import pstats


PROFILE_FILE = "training_profile.prof"


stats = pstats.Stats(PROFILE_FILE)

# stats.strip_dirs()
# stats.sort_stats("cumulative")
stats.sort_stats("tottime")
stats.print_stats("splendor_v1", 30)