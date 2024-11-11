total_timesteps = 5400
number_cars = 1000
shape = 2

src_nodes = ["S-IN", "W-IN", "N-IN", "E-IN"]  # keep the order as SWNE
dst_nodes = ["S-OUT", "W-OUT", "N-OUT", "E-OUT"]  # keep the order as SWNE

src_prob = [0.25, 0.25, 0.25, 0.25]  # keep the order as SWNE
turn_prob = [0, 0.125, 0.75, 0.125]  # keep the order as U turn, left, straight, right

emergency_probs = [0.05, 0.001, 0.05, 0.001] # keep the order as SWNE
