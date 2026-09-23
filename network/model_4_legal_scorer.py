import torch
import torch.nn as nn

from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.env.core.constants import OBSERVATION_SIZE


# ============================================================
# Model 4: Structured self-attention + legal-action scorer + WDL
# ============================================================
#
# Observation input remains unchanged:
#
#     observations:
#         [batch, 258]
#
# Policy input is now the set of candidate legal action IDs:
#
#     legal_action_ids:
#         [batch, num_candidates]
#
#     legal_action_mask:
#         [batch, num_candidates]
#
# Policy output is dynamic:
#
#     legal policy logits:
#         [batch, num_candidates]
#
#     WDL logits:
#         [batch, 3]
#
# Unlike Model 3, Model 4 does NOT emit a flat 1,139-way policy.
# Each legal canonical action ID is embedded and scored against the
# contextualized STATE representation by one shared scoring network.
#
# The number of candidate actions is therefore dynamic. Training
# batches may pad to the largest candidate count in that batch and
# pass legal_action_mask=False for padded entries. There is no global
# 64-action limit.
#
# WDL class order is LOSS, DRAW, WIN. MCTS converts WDL logits to:
#
#     P(WIN) - P(LOSS)
#
# The structured 26-token attention trunk is inherited from Model 3.
# ============================================================


MODEL_DIM = 192
NUM_ATTENTION_HEADS = 4
NUM_ATTENTION_BLOCKS = 3
FEEDFORWARD_MULTIPLIER = 4
DROPOUT = 0.0

# Model 4 policy scorer.
ACTION_EMBED_DIM = 128
ACTION_SCORER_HIDDEN_DIM = 192


# ============================================================
# WDL value layout
# ============================================================


WDL_LOSS = 0
WDL_DRAW = 1
WDL_WIN = 2
WDL_SIZE = 3


# ============================================================
# Observation layout
# ============================================================


class ObservationLayout:
    """
    Exact layout of the current 258-dimensional ObservationEncoder.

    Top-level:
        current player   0:48
        opponent        48:96
        bank            96:102
        decks          102:105
        nobles         105:123
        board          123:255
        node type      255:258
    """

    OBSERVATION_SIZE = 258

    PLAYER_START = 0
    PLAYER_END = 48

    OPPONENT_START = 48
    OPPONENT_END = 96

    BANK_START = 96
    BANK_END = 102

    DECK_START = 102
    DECK_END = 105

    NOBLE_START = 105
    NOBLE_END = 123

    BOARD_START = 123
    BOARD_END = 255

    NODE_TYPE_START = 255
    NODE_TYPE_END = 258

    # Relative layout inside either 48-feature player block.
    PLAYER_GEMS_START = 0
    PLAYER_GEMS_END = 6

    PLAYER_BONUSES_START = 6
    PLAYER_BONUSES_END = 11

    PLAYER_RESERVED_START = 11
    RESERVED_CARD_SIZE = 12
    RESERVED_CARD_COUNT = 3

    PLAYER_POINTS_INDEX = 47

    # Reserved card:
    #   11 normal card features
    #   1 hidden flag
    CARD_SIZE = 11
    RESERVED_HIDDEN_INDEX = 11

    NOBLE_COUNT = 3
    NOBLE_SIZE = 6

    VISIBLE_CARD_COUNT = 12
    VISIBLE_CARD_SIZE = 11

    BOARD_SLOTS_PER_TIER = 4


# ============================================================
# Token indices
# ============================================================


class TokenIndex:
    """
    Fixed token order used by the attention network.

    Keeping this explicit makes future attention diagnostics much
    easier to interpret.
    """

    STATE = 0
    SELF_PLAYER = 1
    OPPONENT = 2
    BANK = 3
    DECK = 4

    NOBLE_START = 5
    NOBLE_END = 8

    VISIBLE_START = 8
    VISIBLE_END = 20

    SELF_RESERVED_START = 20
    SELF_RESERVED_END = 23

    OPP_RESERVED_START = 23
    OPP_RESERVED_END = 26

    COUNT = 26


# ============================================================
# Small shared entity encoders
# ============================================================


class EntityEncoder(nn.Module):
    """
    Small MLP that converts raw normalized entity features into the
    common MODEL_DIM space used by attention.
    """

    def __init__(
        self,
        input_size: int,
        model_dim: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_size,
                model_dim,
            ),
            nn.GELU(),
            nn.Linear(
                model_dim,
                model_dim,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.net(x)


# ============================================================
# Attention block
# ============================================================


class AttentionBlock(nn.Module):
    """
    Pre-norm Transformer-style self-attention block.

    Every valid token may attend to every other valid token.

    Empty visible-card / reserved-card / noble slots are masked as
    keys, so real game objects do not waste attention on padding.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_multiplier: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm_1 = nn.LayerNorm(
            model_dim
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_2 = nn.LayerNorm(
            model_dim
        )

        feedforward_size = (
            model_dim
            * feedforward_multiplier
        )

        self.feedforward = nn.Sequential(
            nn.Linear(
                model_dim,
                feedforward_size,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                feedforward_size,
                model_dim,
            ),
            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):

        normalized = self.norm_1(
            x
        )

        attention_output, attention_weights = (
            self.attention(
                normalized,
                normalized,
                normalized,
                key_padding_mask=key_padding_mask,
                need_weights=return_attention,
                average_attn_weights=False,
            )
        )

        x = x + attention_output

        x = x + self.feedforward(
            self.norm_2(
                x
            )
        )

        if return_attention:
            return (
                x,
                attention_weights,
            )

        return x


# ============================================================
# Main network
# ============================================================


class SplendorNetwork(nn.Module):
    """
    Structured attention model for Splendor.

    The environment / replay-buffer observation format is unchanged.

    Raw observation:
        [batch, 258]

    Internal representation:
        26 entity tokens, each MODEL_DIM wide.

    Final global representation:
        contextualized STATE token.

    Policy inputs:
        legal_action_ids:
            [batch, num_candidates]

        legal_action_mask:
            [batch, num_candidates]
            True  = real legal action
            False = padding

    Outputs:
        legal_policy_logits:
            [batch, num_candidates]

        wdl_logits:
            [batch, 3]

    WDL class order:
        0 = LOSS
        1 = DRAW
        2 = WIN

    The WDL logits are from the current-player perspective.
    """

    def __init__(
        self,
        observation_size: int = OBSERVATION_SIZE,
        action_space_size: int = ACTION_SPACE_SIZE,
        model_dim: int = MODEL_DIM,
        num_attention_heads: int = NUM_ATTENTION_HEADS,
        num_attention_blocks: int = NUM_ATTENTION_BLOCKS,
        feedforward_multiplier: int = FEEDFORWARD_MULTIPLIER,
        dropout: float = DROPOUT,
        action_embedding_dim: int = ACTION_EMBED_DIM,
        action_scorer_hidden_dim: int = ACTION_SCORER_HIDDEN_DIM,
    ):
        super().__init__()

        if (
            observation_size
            != ObservationLayout.OBSERVATION_SIZE
        ):
            raise ValueError(
                "model_4_legal_scorer.py currently expects "
                f"observation_size={ObservationLayout.OBSERVATION_SIZE}, "
                f"got {observation_size}."
            )

        if (
            model_dim
            % num_attention_heads
            != 0
        ):
            raise ValueError(
                "model_dim must be divisible by "
                "num_attention_heads."
            )

        self.observation_size = (
            observation_size
        )

        self.action_space_size = (
            action_space_size
        )

        self.model_dim = (
            model_dim
        )

        self.action_embedding_dim = int(
            action_embedding_dim
        )

        self.action_scorer_hidden_dim = int(
            action_scorer_hidden_dim
        )

        # Canonical action IDs are 0..action_space_size-1.
        # The extra ID is reserved exclusively for batch padding.
        self.padding_action_id = int(
            action_space_size
        )

        # ----------------------------------------------------
        # Shared object encoders
        # ----------------------------------------------------

        self.player_encoder = EntityEncoder(
            input_size=12,
            model_dim=model_dim,
        )

        self.card_encoder = EntityEncoder(
            input_size=11,
            model_dim=model_dim,
        )

        self.noble_encoder = EntityEncoder(
            input_size=6,
            model_dim=model_dim,
        )

        self.bank_encoder = EntityEncoder(
            input_size=6,
            model_dim=model_dim,
        )

        self.deck_encoder = EntityEncoder(
            input_size=3,
            model_dim=model_dim,
        )

        self.node_encoder = EntityEncoder(
            input_size=3,
            model_dim=model_dim,
        )

        # ----------------------------------------------------
        # Learned context embeddings
        # ----------------------------------------------------

        # Token type:
        #   0 STATE
        #   1 PLAYER
        #   2 BANK
        #   3 DECK
        #   4 NOBLE
        #   5 VISIBLE_CARD
        #   6 RESERVED_CARD
        self.token_type_embedding = nn.Embedding(
            7,
            model_dim,
        )

        # Role:
        #   0 none
        #   1 self
        #   2 opponent
        self.role_embedding = nn.Embedding(
            3,
            model_dim,
        )

        # Tier:
        #   0 unknown / not applicable
        #   1 tier 1
        #   2 tier 2
        #   3 tier 3
        self.tier_embedding = nn.Embedding(
            4,
            model_dim,
        )

        # Visible cards use slots 0..3.
        # Reserved cards / nobles use slots 0..2.
        self.slot_embedding = nn.Embedding(
            4,
            model_dim,
        )

        # Opponent hidden reserved cards need to be distinguishable
        # from empty reserved slots even though both have zero card
        # content.
        self.hidden_reserved_embedding = nn.Parameter(
            torch.zeros(
                model_dim
            )
        )

        nn.init.normal_(
            self.hidden_reserved_embedding,
            mean=0.0,
            std=0.02,
        )

        # STATE starts as a learned global vector plus node type.
        self.state_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                model_dim,
            )
        )

        nn.init.normal_(
            self.state_token,
            mean=0.0,
            std=0.02,
        )

        # ----------------------------------------------------
        # Attention trunk
        # ----------------------------------------------------

        self.attention_blocks = nn.ModuleList(
            [
                AttentionBlock(
                    model_dim=model_dim,
                    num_heads=num_attention_heads,
                    feedforward_multiplier=feedforward_multiplier,
                    dropout=dropout,
                )
                for _ in range(
                    num_attention_blocks
                )
            ]
        )

        self.final_norm = nn.LayerNorm(
            model_dim
        )

        # ----------------------------------------------------
        # Heads
        # ----------------------------------------------------

        # Model 4 replaces the flat 1,139-way policy head with a
        # shared legal-action scorer.
        #
        # Valid canonical action IDs:
        #     0 .. action_space_size - 1
        #
        # Padding action ID:
        #     action_space_size
        #
        # The padding embedding is fixed at zero by padding_idx.
        self.action_embedding = nn.Embedding(
            action_space_size + 1,
            self.action_embedding_dim,
            padding_idx=self.padding_action_id,
        )

        self.legal_action_scorer = nn.Sequential(
            nn.Linear(
                model_dim + self.action_embedding_dim,
                self.action_scorer_hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.action_scorer_hidden_dim,
                1,
            ),
        )

        # Raw WDL logits. Do NOT apply softmax here; training should
        # use cross-entropy on these logits for numerical stability.
        self.wdl_head = nn.Linear(
            model_dim,
            WDL_SIZE,
        )

    # ========================================================
    # Observation parsing
    # ========================================================

    def _split_player(
        self,
        player_block: torch.Tensor,
    ):
        """
        player_block:
            [batch, 48]

        Returns:
            resources:
                [batch, 12]
                = gems 6 + bonuses 5 + points 1

            reserved_cards:
                [batch, 3, 11]

            hidden_flags:
                [batch, 3, 1]

            reserved_present:
                [batch, 3] boolean
        """

        gems = player_block[
            :,
            ObservationLayout.PLAYER_GEMS_START:
            ObservationLayout.PLAYER_GEMS_END
        ]

        bonuses = player_block[
            :,
            ObservationLayout.PLAYER_BONUSES_START:
            ObservationLayout.PLAYER_BONUSES_END
        ]

        points = player_block[
            :,
            ObservationLayout.PLAYER_POINTS_INDEX:
            ObservationLayout.PLAYER_POINTS_INDEX + 1
        ]

        resources = torch.cat(
            [
                gems,
                bonuses,
                points,
            ],
            dim=-1,
        )

        reserved_raw = player_block[
            :,
            ObservationLayout.PLAYER_RESERVED_START:
            ObservationLayout.PLAYER_POINTS_INDEX
        ].reshape(
            player_block.shape[0],
            ObservationLayout.RESERVED_CARD_COUNT,
            ObservationLayout.RESERVED_CARD_SIZE,
        )

        reserved_cards = reserved_raw[
            :,
            :,
            :ObservationLayout.CARD_SIZE
        ]

        hidden_flags = reserved_raw[
            :,
            :,
            ObservationLayout.RESERVED_HIDDEN_INDEX:
            ObservationLayout.RESERVED_HIDDEN_INDEX + 1
        ]

        card_has_content = (
            reserved_cards
            .abs()
            .sum(
                dim=-1
            )
            > 0
        )

        card_is_hidden = (
            hidden_flags.squeeze(
                -1
            )
            > 0.5
        )

        reserved_present = (
            card_has_content
            | card_is_hidden
        )

        return (
            resources,
            reserved_cards,
            hidden_flags,
            reserved_present,
        )

    def _parse_observation(
        self,
        x: torch.Tensor,
    ):
        """
        Convert flat [batch, 258] observation into structured tensors.

        This performs NO renormalization and does not alter the encoder
        format. It only rearranges the already-normalized values.
        """

        if x.ndim != 2:
            raise ValueError(
                "Expected observation tensor with shape "
                "[batch, 258]."
            )

        if (
            x.shape[-1]
            != ObservationLayout.OBSERVATION_SIZE
        ):
            raise ValueError(
                "Expected observation size "
                f"{ObservationLayout.OBSERVATION_SIZE}, "
                f"got {x.shape[-1]}."
            )

        self_block = x[
            :,
            ObservationLayout.PLAYER_START:
            ObservationLayout.PLAYER_END
        ]

        opponent_block = x[
            :,
            ObservationLayout.OPPONENT_START:
            ObservationLayout.OPPONENT_END
        ]

        (
            self_resources,
            self_reserved_cards,
            self_reserved_hidden,
            self_reserved_present,
        ) = self._split_player(
            self_block
        )

        (
            opponent_resources,
            opponent_reserved_cards,
            opponent_reserved_hidden,
            opponent_reserved_present,
        ) = self._split_player(
            opponent_block
        )

        bank = x[
            :,
            ObservationLayout.BANK_START:
            ObservationLayout.BANK_END
        ]

        decks = x[
            :,
            ObservationLayout.DECK_START:
            ObservationLayout.DECK_END
        ]

        nobles = x[
            :,
            ObservationLayout.NOBLE_START:
            ObservationLayout.NOBLE_END
        ].reshape(
            x.shape[0],
            ObservationLayout.NOBLE_COUNT,
            ObservationLayout.NOBLE_SIZE,
        )

        visible_cards = x[
            :,
            ObservationLayout.BOARD_START:
            ObservationLayout.BOARD_END
        ].reshape(
            x.shape[0],
            ObservationLayout.VISIBLE_CARD_COUNT,
            ObservationLayout.VISIBLE_CARD_SIZE,
        )

        node_type = x[
            :,
            ObservationLayout.NODE_TYPE_START:
            ObservationLayout.NODE_TYPE_END
        ]

        noble_present = (
            nobles
            .abs()
            .sum(
                dim=-1
            )
            > 0
        )

        visible_present = (
            visible_cards
            .abs()
            .sum(
                dim=-1
            )
            > 0
        )

        return {
            "self_resources":
                self_resources,

            "opponent_resources":
                opponent_resources,

            "self_reserved_cards":
                self_reserved_cards,

            "self_reserved_hidden":
                self_reserved_hidden,

            "self_reserved_present":
                self_reserved_present,

            "opponent_reserved_cards":
                opponent_reserved_cards,

            "opponent_reserved_hidden":
                opponent_reserved_hidden,

            "opponent_reserved_present":
                opponent_reserved_present,

            "bank":
                bank,

            "decks":
                decks,

            "nobles":
                nobles,

            "noble_present":
                noble_present,

            "visible_cards":
                visible_cards,

            "visible_present":
                visible_present,

            "node_type":
                node_type,
        }

    # ========================================================
    # Reserved cards
    # ========================================================

    def _encode_reserved_cards(
        self,
        cards: torch.Tensor,
        hidden_flags: torch.Tensor,
        role_id: int,
    ):
        """
        cards:
            [batch, 3, 11]

        hidden_flags:
            [batch, 3, 1]
        """

        device = cards.device

        encoded = self.card_encoder(
            cards
        )

        reserved_type = (
            self.token_type_embedding(
                torch.tensor(
                    6,
                    dtype=torch.long,
                    device=device,
                )
            )
            .view(
                1,
                1,
                -1,
            )
        )

        role = (
            self.role_embedding(
                torch.tensor(
                    role_id,
                    dtype=torch.long,
                    device=device,
                )
            )
            .view(
                1,
                1,
                -1,
            )
        )

        slots = (
            self.slot_embedding(
                torch.arange(
                    ObservationLayout.RESERVED_CARD_COUNT,
                    device=device,
                )
            )
            .unsqueeze(
                0
            )
        )

        hidden_context = (
            hidden_flags
            * self.hidden_reserved_embedding.view(
                1,
                1,
                -1,
            )
        )

        return (
            encoded
            + reserved_type
            + role
            + slots
            + hidden_context
        )

    # ========================================================
    # Token construction
    # ========================================================

    def _build_tokens(
        self,
        x: torch.Tensor,
    ):
        """
        Returns:
            tokens:
                [batch, 26, model_dim]

            key_padding_mask:
                [batch, 26]

        key_padding_mask:
            False = real token
            True  = empty/padded token
        """

        parsed = self._parse_observation(
            x
        )

        batch_size = x.shape[0]
        device = x.device

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        state = self.state_token.expand(
            batch_size,
            -1,
            -1,
        ).squeeze(
            1
        )

        state = (
            state
            + self.node_encoder(
                parsed[
                    "node_type"
                ]
            )
            + self.token_type_embedding(
                torch.zeros(
                    batch_size,
                    dtype=torch.long,
                    device=device,
                )
            )
        )

        # ----------------------------------------------------
        # Players
        # ----------------------------------------------------

        player_type = self.token_type_embedding(
            torch.full(
                (
                    batch_size,
                ),
                1,
                dtype=torch.long,
                device=device,
            )
        )

        self_player = (
            self.player_encoder(
                parsed[
                    "self_resources"
                ]
            )
            + player_type
            + self.role_embedding(
                torch.full(
                    (
                        batch_size,
                    ),
                    1,
                    dtype=torch.long,
                    device=device,
                )
            )
        )

        opponent = (
            self.player_encoder(
                parsed[
                    "opponent_resources"
                ]
            )
            + player_type
            + self.role_embedding(
                torch.full(
                    (
                        batch_size,
                    ),
                    2,
                    dtype=torch.long,
                    device=device,
                )
            )
        )

        # ----------------------------------------------------
        # Bank
        # ----------------------------------------------------

        bank = (
            self.bank_encoder(
                parsed[
                    "bank"
                ]
            )
            + self.token_type_embedding(
                torch.full(
                    (
                        batch_size,
                    ),
                    2,
                    dtype=torch.long,
                    device=device,
                )
            )
        )

        # ----------------------------------------------------
        # Deck
        # ----------------------------------------------------

        deck = (
            self.deck_encoder(
                parsed[
                    "decks"
                ]
            )
            + self.token_type_embedding(
                torch.full(
                    (
                        batch_size,
                    ),
                    3,
                    dtype=torch.long,
                    device=device,
                )
            )
        )

        # ----------------------------------------------------
        # Nobles
        # ----------------------------------------------------

        nobles = self.noble_encoder(
            parsed[
                "nobles"
            ]
        )

        noble_type = (
            self.token_type_embedding(
                torch.tensor(
                    4,
                    dtype=torch.long,
                    device=device,
                )
            )
            .view(
                1,
                1,
                -1,
            )
        )

        noble_slots = (
            self.slot_embedding(
                torch.arange(
                    ObservationLayout.NOBLE_COUNT,
                    device=device,
                )
            )
            .unsqueeze(
                0
            )
        )

        nobles = (
            nobles
            + noble_type
            + noble_slots
        )

        # ----------------------------------------------------
        # Visible cards
        # ----------------------------------------------------

        visible_cards = self.card_encoder(
            parsed[
                "visible_cards"
            ]
        )

        visible_type = (
            self.token_type_embedding(
                torch.tensor(
                    5,
                    dtype=torch.long,
                    device=device,
                )
            )
            .view(
                1,
                1,
                -1,
            )
        )

        visible_tiers = torch.tensor(
            [
                1, 1, 1, 1,
                2, 2, 2, 2,
                3, 3, 3, 3,
            ],
            dtype=torch.long,
            device=device,
        )

        visible_slots = torch.tensor(
            [
                0, 1, 2, 3,
                0, 1, 2, 3,
                0, 1, 2, 3,
            ],
            dtype=torch.long,
            device=device,
        )

        visible_cards = (
            visible_cards
            + visible_type
            + self.tier_embedding(
                visible_tiers
            ).unsqueeze(
                0
            )
            + self.slot_embedding(
                visible_slots
            ).unsqueeze(
                0
            )
        )

        # ----------------------------------------------------
        # Reserved cards
        # ----------------------------------------------------

        self_reserved = self._encode_reserved_cards(
            cards=parsed[
                "self_reserved_cards"
            ],
            hidden_flags=parsed[
                "self_reserved_hidden"
            ],
            role_id=1,
        )

        opponent_reserved = self._encode_reserved_cards(
            cards=parsed[
                "opponent_reserved_cards"
            ],
            hidden_flags=parsed[
                "opponent_reserved_hidden"
            ],
            role_id=2,
        )

        # ----------------------------------------------------
        # Sequence
        # ----------------------------------------------------

        tokens = torch.cat(
            [
                state.unsqueeze(
                    1
                ),

                self_player.unsqueeze(
                    1
                ),

                opponent.unsqueeze(
                    1
                ),

                bank.unsqueeze(
                    1
                ),

                deck.unsqueeze(
                    1
                ),

                nobles,

                visible_cards,

                self_reserved,

                opponent_reserved,
            ],
            dim=1,
        )

        if (
            tokens.shape[1]
            != TokenIndex.COUNT
        ):
            raise RuntimeError(
                "Unexpected token count: "
                f"{tokens.shape[1]}"
            )

        # ----------------------------------------------------
        # Empty-token mask
        # ----------------------------------------------------

        always_present = torch.zeros(
            (
                batch_size,
                5,
            ),
            dtype=torch.bool,
            device=device,
        )

        key_padding_mask = torch.cat(
            [
                always_present,

                ~parsed[
                    "noble_present"
                ],

                ~parsed[
                    "visible_present"
                ],

                ~parsed[
                    "self_reserved_present"
                ],

                ~parsed[
                    "opponent_reserved_present"
                ],
            ],
            dim=1,
        )

        return (
            tokens,
            key_padding_mask,
        )

    # ========================================================
    # Attention trunk
    # ========================================================

    def _forward_features(
        self,
        x: torch.Tensor,
    ):
        tokens, key_padding_mask = (
            self._build_tokens(
                x
            )
        )

        for block in self.attention_blocks:
            tokens = block(
                tokens,
                key_padding_mask=key_padding_mask,
            )

        tokens = self.final_norm(
            tokens
        )

        # STATE token = learned whole-board representation.
        state_features = tokens[
            :,
            TokenIndex.STATE,
            :
        ]

        return state_features

    # ========================================================
    # WDL helpers
    # ========================================================

    @staticmethod
    def wdl_probabilities(
        wdl_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert raw LOSS / DRAW / WIN logits to probabilities.

        Input:
            [..., 3]

        Output:
            [..., 3]
        """

        if (
            wdl_logits.shape[-1]
            != WDL_SIZE
        ):
            raise ValueError(
                "Expected WDL logits with final dimension "
                f"{WDL_SIZE}, got {wdl_logits.shape[-1]}."
            )

        return torch.softmax(
            wdl_logits,
            dim=-1,
        )

    @staticmethod
    def wdl_logits_to_value(
        wdl_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert WDL logits to the scalar value used by MCTS.

        value = P(WIN) - P(LOSS)

        Draw contributes zero. The returned tensor keeps a final
        singleton dimension so [batch, 3] becomes [batch, 1].
        """

        probabilities = (
            SplendorNetwork.wdl_probabilities(
                wdl_logits
            )
        )

        value = (
            probabilities[..., WDL_WIN]
            - probabilities[..., WDL_LOSS]
        )

        return value.unsqueeze(
            -1
        )

    # ========================================================
    # Legal-action input preparation
    # ========================================================

    def _prepare_legal_action_inputs(
        self,
        legal_action_ids: torch.Tensor,
        legal_action_mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ):
        """
        Normalize legal-action tensors for both MCTS and training.

        Supported legal_action_ids shapes:

            [num_candidates]
                Single-state MCTS convenience form.
                Requires batch_size == 1.

            [batch, num_candidates]
                Batched training / evaluation form.

        Padding:
            self.padding_action_id == ACTION_SPACE_SIZE

        If legal_action_mask is omitted for a 2D padded batch, it is
        inferred from legal_action_ids != self.padding_action_id.
        """

        legal_action_ids = torch.as_tensor(
            legal_action_ids,
            dtype=torch.long,
            device=device,
        )

        if legal_action_ids.ndim == 1:

            if batch_size != 1:
                raise ValueError(
                    "1D legal_action_ids may only be used "
                    "with a single observation."
                )

            legal_action_ids = (
                legal_action_ids.unsqueeze(
                    0
                )
            )

        elif legal_action_ids.ndim != 2:

            raise ValueError(
                "legal_action_ids must have shape "
                "[num_candidates] or "
                "[batch, num_candidates]."
            )

        if (
            legal_action_ids.shape[0]
            != batch_size
        ):

            raise ValueError(
                "legal_action_ids batch dimension "
                "does not match observations: "
                f"{legal_action_ids.shape[0]} vs "
                f"{batch_size}."
            )

        if legal_action_ids.shape[1] == 0:

            raise ValueError(
                "Each position must provide at least "
                "one candidate action."
            )

        # Embedding lookup must always remain in range.
        if torch.any(
            legal_action_ids < 0
        ) or torch.any(
            legal_action_ids
            > self.padding_action_id
        ):

            raise ValueError(
                "legal_action_ids contains an ID "
                "outside the canonical action space "
                "or padding ID."
            )

        if legal_action_mask is None:

            legal_action_mask = (
                legal_action_ids
                != self.padding_action_id
            )

        else:

            legal_action_mask = torch.as_tensor(
                legal_action_mask,
                dtype=torch.bool,
                device=device,
            )

            if (
                legal_action_mask.shape
                != legal_action_ids.shape
            ):

                raise ValueError(
                    "legal_action_mask must have "
                    "the same shape as "
                    "legal_action_ids."
                )

        # A padding ID may never be marked as a real action.
        if torch.any(
            legal_action_mask
            & (
                legal_action_ids
                == self.padding_action_id
            )
        ):

            raise ValueError(
                "Padding action ID was marked as "
                "a real legal action."
            )

        # Every state must have at least one real candidate.
        if not torch.all(
            legal_action_mask.any(
                dim=1
            )
        ):

            raise ValueError(
                "Every position must contain at "
                "least one real legal action."
            )

        return (
            legal_action_ids,
            legal_action_mask,
        )

    # ========================================================
    # Shared legal-action scorer
    # ========================================================

    def _score_legal_actions(
        self,
        state_features: torch.Tensor,
        legal_action_ids: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Score only the candidate legal actions for each state.

        state_features:
            [batch, model_dim]

        legal_action_ids:
            [num_candidates]
            or
            [batch, num_candidates]

        Returns:
            legal_logits:
                [batch, num_candidates]

        Padded positions are assigned the smallest finite value for
        the current dtype. This makes their softmax probability
        effectively zero while avoiding 0 * -inf NaNs in a
        probability-target cross-entropy implementation.
        """

        (
            legal_action_ids,
            legal_action_mask,
        ) = self._prepare_legal_action_inputs(
            legal_action_ids=legal_action_ids,
            legal_action_mask=legal_action_mask,
            batch_size=state_features.shape[0],
            device=state_features.device,
        )

        action_features = (
            self.action_embedding(
                legal_action_ids
            )
        )

        num_candidates = (
            legal_action_ids.shape[1]
        )

        expanded_state = (
            state_features
            .unsqueeze(
                1
            )
            .expand(
                -1,
                num_candidates,
                -1,
            )
        )

        scorer_input = torch.cat(
            [
                expanded_state,
                action_features,
            ],
            dim=-1,
        )

        legal_logits = (
            self.legal_action_scorer(
                scorer_input
            )
            .squeeze(
                -1
            )
        )

        masked_value = torch.finfo(
            legal_logits.dtype
        ).min

        legal_logits = (
            legal_logits.masked_fill(
                ~legal_action_mask,
                masked_value,
            )
        )

        return legal_logits

    # ========================================================
    # WDL-only forward
    # ========================================================

    def forward_wdl(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute only the WDL logits.

        Useful for value diagnostics or WDL-only training phases.
        """

        features = self._forward_features(
            x
        )

        return self.wdl_head(
            features
        )

    # ========================================================
    # Standard Model 4 forward
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
        legal_action_ids: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
    ):
        """
        Model 4 requires candidate legal actions.

        Returns:
            legal_policy_logits:
                [batch, num_candidates]

            wdl_logits:
                [batch, 3]
        """

        features = self._forward_features(
            x
        )

        legal_logits = (
            self._score_legal_actions(
                state_features=features,
                legal_action_ids=legal_action_ids,
                legal_action_mask=(
                    legal_action_mask
                ),
            )
        )

        wdl_logits = self.wdl_head(
            features
        )

        return (
            legal_logits,
            wdl_logits,
        )

    # ========================================================
    # MCTS-friendly legal forward
    # ========================================================

    def forward_legal(
        self,
        x: torch.Tensor,
        legal_action_ids: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
    ):
        """
        Alias for the Model 4 legal-action policy interface.

        For MCTS, x is normally [1, 258] and legal_action_ids may be
        a simple 1D tensor [num_legal_actions].

        The returned policy tensor remains 2D:
            [1, num_legal_actions]

        so the neural evaluator interface stays straightforward.
        """

        return self.forward(
            x=x,
            legal_action_ids=legal_action_ids,
            legal_action_mask=legal_action_mask,
        )

    # ========================================================
    # Attention diagnostics
    # ========================================================

    def forward_with_attention(
        self,
        x: torch.Tensor,
        legal_action_ids: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
    ):
        """
        Diagnostic forward pass.

        Returns:
            legal_policy_logits
            wdl_logits
            attention_maps

        Each attention map has shape:
            [batch, heads, 26, 26]

        Token order is documented in TokenIndex.
        """

        tokens, key_padding_mask = (
            self._build_tokens(
                x
            )
        )

        attention_maps = []

        for block in self.attention_blocks:
            (
                tokens,
                attention_weights,
            ) = block(
                tokens,
                key_padding_mask=key_padding_mask,
                return_attention=True,
            )

            attention_maps.append(
                attention_weights
            )

        tokens = self.final_norm(
            tokens
        )

        features = tokens[
            :,
            TokenIndex.STATE,
            :
        ]

        legal_logits = (
            self._score_legal_actions(
                state_features=features,
                legal_action_ids=legal_action_ids,
                legal_action_mask=(
                    legal_action_mask
                ),
            )
        )

        wdl_logits = self.wdl_head(
            features
        )

        return (
            legal_logits,
            wdl_logits,
            attention_maps,
        )
