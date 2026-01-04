import itertools

# ranks 2-a: 0-12
# suits s-c: 0-3
# card: rank*4 + suit

def get_rank_suit(card):
    return card // 4, card % 4

def evaluate_5_cards(cards):
    # scoring
    ranks = sorted([c // 4 for c in cards], reverse=True)
    suits = [c % 4 for c in cards]
    
    # rank counts
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
        
    sorted_counts = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    is_flush = len(set(suits)) == 1
    
    # straight
    unique_ranks = []
    for r in ranks:
        if not unique_ranks or r != unique_ranks[-1]:
            unique_ranks.append(r)
            
    is_straight = False
    straight_high = -1
    if len(unique_ranks) >= 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        elif unique_ranks[0] == 12 and unique_ranks[1] == 3 and unique_ranks[4] == 0:
            is_straight = True
            straight_high = 3 

    # logic
    if is_flush and is_straight: return (8, straight_high)
    if sorted_counts[0][1] == 4: return (7, sorted_counts[0][0], sorted_counts[1][0])
    if sorted_counts[0][1] == 3 and sorted_counts[1][1] == 2: return (6, sorted_counts[0][0], sorted_counts[1][0])
    if is_flush: return (5, ranks)
    if is_straight: return (4, straight_high)
    if sorted_counts[0][1] == 3: return (3, sorted_counts[0][0], [r for r, c in sorted_counts[1:]])
    if sorted_counts[0][1] == 2 and sorted_counts[1][1] == 2: return (2, sorted_counts[0][0], sorted_counts[1][0], sorted_counts[2][0])
    if sorted_counts[0][1] == 2: return (1, sorted_counts[0][0], [r for r, c in sorted_counts[1:]])
    return (0, ranks)

def evaluate_7_cards(cards):
    best = (-1,)
    # itertools faster than manual python loops
    for combo in itertools.combinations(cards, 5):
        score = evaluate_5_cards(combo)
        if score > best:
            best = score
    return best
