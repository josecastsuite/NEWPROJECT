#include "josecast/gating_tree.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>

namespace josecast {

nb::tuple solve_gating_times(
    int32_t source_bidx,
    double source_q,
    nb::ndarray<nb::numpy, double, nb::shape<-1>> body_volumes,
    nb::ndarray<nb::numpy, int32_t, nb::shape<-1, 2>> edges,
    nb::ndarray<nb::numpy, double, nb::shape<-1>> edge_q,
    nb::ndarray<nb::numpy, double, nb::shape<-1>> q_in)
{
    const size_t n = body_volumes.shape(0);
    const size_t E = edges.shape(0);

    std::vector<double> vol(body_volumes.data(), body_volumes.data() + n);
    std::vector<double> Q(n, 0.0);
    std::vector<double> t_enter(n, std::numeric_limits<double>::infinity());
    std::vector<double> t_exit(n, std::numeric_limits<double>::infinity());

    std::vector<std::vector<std::pair<int32_t, double>>> children(n);
    std::vector<std::vector<std::pair<int32_t, double>>> parents(n);

    Q[static_cast<size_t>(source_bidx)] = source_q;

    const int32_t* e_data = edges.data();
    const double* qe_data = edge_q.data();
    for (size_t i = 0; i < E; ++i) {
        int32_t p = e_data[2 * i + 0];
        int32_t c = e_data[2 * i + 1];
        if (p < 0 || p >= static_cast<int32_t>(n) || c < 0 || c >= static_cast<int32_t>(n)) {
            continue;
        }
        double q = qe_data[i];
        children[static_cast<size_t>(p)].push_back({c, q});
        parents[static_cast<size_t>(c)].push_back({p, q});
        Q[static_cast<size_t>(c)] += q;
    }

    if (q_in.shape(0) == static_cast<ptrdiff_t>(n)) {
        const double* qin_data = q_in.data();
        for (size_t i = 0; i < n; ++i) {
            Q[i] = std::max(Q[i], qin_data[i]);
        }
    }

    // Reachability from the source body.
    std::vector<char> reachable(n, 0);
    std::queue<int32_t> bfs;
    reachable[static_cast<size_t>(source_bidx)] = 1;
    bfs.push(source_bidx);
    while (!bfs.empty()) {
        int32_t u = bfs.front();
        bfs.pop();
        for (const auto& ch : children[static_cast<size_t>(u)]) {
            if (!reachable[static_cast<size_t>(ch.first)]) {
                reachable[static_cast<size_t>(ch.first)] = 1;
                bfs.push(ch.first);
            }
        }
    }

    // Kahn topological sort on the reachable sub-graph.  Nodes with several
    // parents only become ready after all their parents have been processed,
    // so t_enter is correctly the maximum parent t_exit.
    std::vector<int32_t> indeg(n, 0);
    for (size_t u = 0; u < n; ++u) {
        if (!reachable[u]) continue;
        for (const auto& ch : children[u]) {
            if (reachable[static_cast<size_t>(ch.first)]) {
                indeg[static_cast<size_t>(ch.first)]++;
            }
        }
    }

    std::queue<int32_t> q;
    for (size_t u = 0; u < n; ++u) {
        if (reachable[u] && indeg[u] == 0) {
            q.push(static_cast<int32_t>(u));
        }
    }

    t_enter[static_cast<size_t>(source_bidx)] = 0.0;

    while (!q.empty()) {
        int32_t u = q.front();
        q.pop();
        const size_t ui = static_cast<size_t>(u);

        if (u != source_bidx) {
            double max_te = 0.0;
            bool has_parent = false;
            for (const auto& pr : parents[ui]) {
                int32_t p = pr.first;
                size_t pi = static_cast<size_t>(p);
                if (reachable[pi] && std::isfinite(t_exit[pi])) {
                    if (!has_parent) {
                        max_te = t_exit[pi];
                        has_parent = true;
                    } else {
                        max_te = std::max(max_te, t_exit[pi]);
                    }
                }
            }
            t_enter[ui] = has_parent ? max_te : 0.0;
        }

        t_exit[ui] = t_enter[ui] + vol[ui] / std::max(Q[ui], 1e-18);

        for (const auto& ch : children[ui]) {
            size_t ci = static_cast<size_t>(ch.first);
            if (reachable[ci] && --indeg[ci] == 0) {
                q.push(ch.first);
            }
        }
    }

    // Warn about cycles or disconnected reachable nodes that Kahn could not process.
    for (size_t u = 0; u < n; ++u) {
        if (reachable[u] && !std::isfinite(t_enter[u])) {
            std::cerr << "[josecast_core] gating_tree: cycle detected around node "
                      << u << "; graph has a loop.\n";
        }
    }

    auto* tenter_vec = new std::vector<double>(std::move(t_enter));
    auto* texit_vec = new std::vector<double>(std::move(t_exit));
    auto* Q_vec = new std::vector<double>(std::move(Q));

    auto cap_enter = nb::capsule(tenter_vec, [](void* p) noexcept {
        delete static_cast<std::vector<double>*>(p);
    });
    auto cap_exit = nb::capsule(texit_vec, [](void* p) noexcept {
        delete static_cast<std::vector<double>*>(p);
    });
    auto cap_Q = nb::capsule(Q_vec, [](void* p) noexcept {
        delete static_cast<std::vector<double>*>(p);
    });

    nb::ndarray<nb::numpy, double, nb::shape<-1>> arr_enter(
        tenter_vec->data(), {n}, cap_enter);
    nb::ndarray<nb::numpy, double, nb::shape<-1>> arr_exit(
        texit_vec->data(), {n}, cap_exit);
    nb::ndarray<nb::numpy, double, nb::shape<-1>> arr_Q(
        Q_vec->data(), {n}, cap_Q);

    return nb::make_tuple(arr_enter, arr_exit, arr_Q);
}

} // namespace josecast
