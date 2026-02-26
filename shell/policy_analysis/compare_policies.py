import pickle

with open('policy.pkl','rb') as f:
    policies = pickle.load(f)

n_agents = len(policies)
print('n_agents =', n_agents)
for i, pol in enumerate(policies):
    print(f'agent={i} | n_states={len(pol)}')

# compute shared keys
keys0 = set(policies[0].keys())
for i in range(1, n_agents):
    keysi = set(policies[i].keys())
    shared = keys0 & keysi
    print(f'shared states between 0 and {i}: {len(shared)}')
    # among shared, count identical action assignments
    identical = 0
    diffs = []
    for s in list(shared):
        a0 = policies[0][s]
        ai = policies[i][s]
        if a0 == ai:
            identical += 1
        else:
            diffs.append((s, a0, ai))
    print(f'  identical assignments: {identical} / {len(shared)}')
    print('  sample differences (up to 5):')
    for d in diffs[:5]:
        print('   ', d)

# overall pairwise comparison for all agents
for i in range(n_agents):
    for j in range(i+1, n_agents):
        s_i = set(policies[i].keys())
        s_j = set(policies[j].keys())
        shared = s_i & s_j
        same = sum(1 for s in shared if policies[i][s] == policies[j][s])
        print(f'pair {i}-{j}: shared={len(shared)}, same_action={same}')
