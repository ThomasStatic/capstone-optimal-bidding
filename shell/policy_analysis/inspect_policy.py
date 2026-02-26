import pickle

with open('policy.pkl','rb') as f:
    policies = pickle.load(f)

print('n_agents =', len(policies))
for i, pol in enumerate(policies):
    keys = list(pol.keys())
    print(f'agent={i} | n_states={len(pol)} | sample (first 3) = {keys[:3]}')
