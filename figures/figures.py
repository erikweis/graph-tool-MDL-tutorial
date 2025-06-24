import graph_tool.all as gt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec


def plot_model_complexity_tradeoff():

    g = gt.collection.data['football']

    data = []
    for B in range(1,g.num_vertices()):
        for _ in range(100):
            state_max = gt.minimize_blockmodel_dl(g, multilevel_mcmc_args=dict(B_max = B, merge_sweeps=100))
            state_min = gt.minimize_blockmodel_dl(g, multilevel_mcmc_args=dict(B_min = B, merge_sweeps=100))
            B_max_actual = state_max.get_nonempty_B()
            B_min_actual = state_min.get_nonempty_B()
            if B_max_actual == B:
                print("max worked")
                state = state_max
                break
            elif B_min_actual == B:
                print("min worked")
                state = state_min
                break
            else:
                
                continue
        
        if state.get_nonempty_B() != B:
            print("Failed",B, B_max_actual, B_min_actual)
            continue

        #S = state.entropy(partition_dl=False, degree_dl = False, adjacency=False, edges_dl=False)
        S_params = state.entropy(adjacency=False)
        S_likelihood = state.entropy() - S_params
        B_actual = state.get_nonempty_B()
        print(B, S_params, S_likelihood)
        data.append(dict(B=B,S_params = S_params, S_likelihood=S_likelihood))

    b = g.new_vertex_property("int", vals = range(g.num_vertices()))
    state = gt.BlockState(g, b=b)
    S_params = state.entropy(adjacency=False)
    S_likelihood = state.entropy() - S_params
    B_actual = state.get_nonempty_B()
    data.append(dict(B=B,S_params = S_params, S_likelihood=S_likelihood))

    df = pd.DataFrame(data)
    df['total'] = df.S_params + df.S_likelihood
    df['total'] = df.S_params + df.S_likelihood
    B_star = df[df.total == df.total.min()].B.values[0]

    fig = plt.figure(figsize=(4.5,3.5))
    #plt.axvline(x=1, color = '0.6', linestyle='--')
    plt.axvline(x=B_star, color = '0.6', linestyle='--')
    #plt.axvline(x=g.num_vertices(), color = '0.6', linestyle='--')
    plt.plot(df.B, df.S_params, label = 'Params', color = '#80A4CE', lw = 2)
    plt.plot(df.B, df.S_likelihood, label = 'Data' , color = '#F48A64', lw = 2)
    plt.plot(df.B, df.S_params + df.S_likelihood, label = "Total DL", color = '0.4', lw = 2)

    #plt.xscale('log')
    plt.legend()
    plt.xlabel('Number of groups ($B$)')
    plt.ylabel('Description length (nats)')
    plt.tight_layout()
    plt.savefig('model_complexity_tradeoff.pdf')


def plot_model_complexity_snapshots():
    # switch to a non-interactive backend
    plt.switch_backend('cairo')
    np.random.seed(1)
    gt.seed_rng(3)

    # load graph and compute optimal block state
    g = gt.collection.data['football']
    state = gt.minimize_blockmodel_dl(g)
    best_b = state.b.a
    best_S = state.entropy()
    for _ in range(1000):
        state.mcmc_sweep()
        if state.entropy() < best_S:
            best_S = state.entropy()
            best_b = state.b.a
    
    optimal_state = gt.BlockState(g, b=best_b)
    pos = gt.sfdp_layout(g, groups=optimal_state.b, gamma=.04)

    # define the three complexity states
    states = []
    # underfit: all vertices in one block
    b_under = g.new_vertex_property('int', vals=[0] * g.num_vertices())
    states.append(('Underfit', gt.BlockState(g, b_under)))
    # optimal: as found
    states.append(('Optimal', optimal_state))
    # overfit: each vertex its own block
    b_over = g.new_vertex_property('int', vals=list(range(g.num_vertices())))
    states.append(('Overfit', gt.BlockState(g, b_over)))

    # set up figure with GridSpec: 1 row, 6 columns (bar, graph) x3
    fig = plt.figure(figsize=(12, 4))
    r = 4
    gs = GridSpec(1, 6, figure=fig, width_ratios=[1, r, 1, r, 1, r], wspace=0.2)

    for i, (label, state) in enumerate(states):
        # bar chart axis
        ax_bar = fig.add_subplot(gs[0, 2*i])
        # graph drawing axis
        ax_graph = fig.add_subplot(gs[0, 2*i + 1])

        # calculate block sizes for bar chart
        S_likelihood = state.entropy() - state.entropy(adjacency=False)
        S_params = state.entropy() - S_likelihood
        S_total = state.entropy()

        ax_bar.bar(['Parameters','Data','Total'],[S_params,S_likelihood,S_total],
                   color=['#80A4CE', '#F48A64', '0.4'], width=0.6, edgecolor='black'    )
        plt.sca(ax_bar)
        plt.ylim(0,2600)
        plt.xticks(rotation=45,ha='right')
        #ax_bar.margins(x=0.1)
        ax_bar.set_ylabel('Description length (nats)')

        # draw the graph
        drawer = state.draw(pos=pos, mplfig=ax_graph, vertex_size=1.1)
        drawer.fit_view(yflip=True)
        ax_graph.axis('off')
        xlim = ax_graph.get_xlim()
        # shift left by x%
        pct_shift = -0.06
        shift = (xlim[1] - xlim[0]) * pct_shift
        ax_graph.set_xlim(xlim[0] - shift, xlim[1] - shift)

    # add vertical padding
    plt.subplots_adjust(bottom=0.2,left=0.07,right=1.0)

    plt.savefig('complexity_tradeoff_snapshots.pdf')


if __name__ == '__main__':

    plot_model_complexity_snapshots()

