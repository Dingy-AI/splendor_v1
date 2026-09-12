import torch
import numpy as np

from splendor_v1.network.model import SplendorNetwork


class RawNetworkAgent:

    def __init__(
        self,
        model_path,
        device=None,
    ):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.model = SplendorNetwork().to(
            self.device
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()


    @torch.no_grad()
    def select_action(
        self,
        env,
        state,
    ):

        legal_actions = env._legal_actions(
            state
        )

        if not legal_actions:
            return None

        # Use the SAME encoder used during training.
        observation = env.observation_encoder.encoder(
            state
        )

        observation = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        # --------------------------------
        # Raw network prediction
        # --------------------------------

        policy_logits, value = self.model(
            observation
        )

        policy_logits = (
            policy_logits
            .squeeze(0)
            .cpu()
            .numpy()
        )

        # --------------------------------
        # Only consider legal actions
        # --------------------------------

        legal_ids = np.array([
            env.action_to_id(action)
            for action in legal_actions
        ])

        legal_logits = policy_logits[
            legal_ids
        ]

        best_index = np.argmax(
            legal_logits
        )

        return legal_actions[best_index]