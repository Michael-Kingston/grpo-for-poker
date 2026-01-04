import itertools

# ranks 2-a: 0-12
# suits s-c: 0-3
# card: rank*4 + suit

def get_rank_suit(card):
    return card // 4, card % 4

def evaluate_5_cards(cards):
    # scoring
    ranks = sorted([get_rank_suit(c)[0] for c in cards], reverse=True)
    suits = [get_rank_suit(c)[1] for c in cards]
    
    counts = {r: ranks.count(r) for r in set(ranks)}
    sorted_counts = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    
    is_flush = len(set(suits)) == 1
    
    # straight
    unique_ranks = sorted(list(set(ranks)), reverse=True)
    is_straight = False
    straight_high = -1
    if len(unique_ranks) >= 5:
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i+4] == 4:
                is_straight = True
                straight_high = unique_ranks[i]
                break
        if not is_straight and set([12, 0, 1, 2, 3]).issubset(set(unique_ranks)):
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
    # best 5
    best = (-1,)
    for combo in itertools.combinations(cards, 5):
        score = evaluate_5_cards(combo)
        if score > best:
            best = score
    return best
