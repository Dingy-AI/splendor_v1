# There should be a variable called gold_payment when you do the buy.
# We will be using a heuristic for this. Sigh! :'(

# The idea here is that the payment will be 'discounted' by the bonuses from 
# the card first. Then we will determine the gold usage!

# 0 Gold 
# (0,0,0,0,0)
# 1 possibility -> no gold usage 

# 1 Gold 
# (1,0,0,0,0)
# (0,1,0,0,0)
# (0,0,1,0,0)
# (0,0,0,1,0)
# (0,0,0,0,1)
# 5 possibility max -> 1 gold usage

# 2 Gold (W,B,G,Bl,R)
# WW, BB, GG, BlBl, RR

# WB, WG, WBl, WR... 
# 15 Possibility max ->  2 gold usage


# Gold	Patterns
# 0     	1
# 1   	5
# 2	    15
# 3   	35
# 4	    70
# 5   	126


# buy_id = (
    # buy_offset
    # + card_location * 252
    # + gold_payment_id
# )


COLORS = 5  # White, Blue, Green, Red, Black
MAX_GOLD = 5


def generate_gold_payment_patterns():
    patterns = []

    def recurse(color_index, remaining_gold, current):
        # Last color
        if color_index == COLORS - 1:
            # Assign all remaining gold to this color
            patterns.append(tuple(current + [remaining_gold]))
            return

        # Try assigning 0 -> remaining_gold gold to this color
        for amount in range(remaining_gold + 1):
            recurse(
                color_index + 1,
                remaining_gold - amount,
                current + [amount],
            )

    # Generate exactly 0-5 gold usage
    for gold_used in range(MAX_GOLD + 1):
        recurse(
            0,
            gold_used,
            []
        )

    return patterns