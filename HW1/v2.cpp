#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <climits>

using namespace std;

static const int MAX_ITEM = 1000;

struct FPNode {
    int item;
    int count;
    FPNode* parent;
    unordered_map<int, FPNode*> children;
    FPNode* next;

    FPNode(int item_, FPNode* parent_) : item(item_), count(0), parent(parent_), next(nullptr) {}
};

struct FPTree {
    FPNode* root;
    vector<FPNode*> header_head;
    vector<FPNode*> header_tail;
    vector<int> freq;

    FPTree() : header_head(MAX_ITEM, nullptr), header_tail(MAX_ITEM, nullptr), freq(MAX_ITEM, 0) {
        root = new FPNode(-1, nullptr);
    }

    ~FPTree() {
        destroy(root);
    }

    void destroy(FPNode* node) {
        if (!node) return;
        for (auto& kv : node->children) {
            destroy(kv.second);
        }
        delete node;
    }

    void add_transaction(const vector<int>& items, int cnt = 1) {
        FPNode* cur = root;
        for (int item : items) {
            FPNode* child;
            auto it = cur->children.find(item);
            if (it == cur->children.end()) {
                child = new FPNode(item, cur);
                cur->children[item] = child;

                if (header_head[item] == nullptr) {
                    header_head[item] = child;
                    header_tail[item] = child;
                } else {
                    header_tail[item]->next = child;
                    header_tail[item] = child;
                }
            } else {
                child = it->second;
            }

            child->count += cnt;
            freq[item] += cnt;
            cur = child;
        }
    }

    bool empty() const {
        return root->children.empty();
    }

    bool has_single_path() const {
        FPNode* cur = root;
        while (cur) {
            if (cur->children.size() > 1) return false;
            if (cur->children.empty()) break;
            cur = cur->children.begin()->second;
        }
        return true;
    }
};

vector<vector<int>> read_transactions_fast(const string& filename) {
    ifstream fin(filename);
    if (!fin) {
        throw runtime_error("Cannot open input file: " + filename);
    }

    vector<vector<int>> transactions;
    string line;

    while (getline(fin, line)) {
        if (line.empty()) continue;

        vector<int> trans;
        trans.reserve(32);

        int num = 0;
        bool in_num = false;

        for (char c : line) {
            if (c >= '0' && c <= '9') {
                num = num * 10 + (c - '0');
                in_num = true;
            } else if (c == ',') {
                if (in_num) {
                    trans.push_back(num);
                    num = 0;
                    in_num = false;
                }
            }
        }
        if (in_num) trans.push_back(num);

        sort(trans.begin(), trans.end());
        trans.erase(unique(trans.begin(), trans.end()), trans.end());

        transactions.push_back(std::move(trans));
    }

    return transactions;
}

FPTree* build_fp_tree(const vector<vector<int>>& transactions, int min_count) {
    vector<int> global_freq(MAX_ITEM, 0);

    for (const auto& trans : transactions) {
        for (int item : trans) {
            ++global_freq[item];
        }
    }

    FPTree* tree = new FPTree();

    for (const auto& trans : transactions) {
        vector<int> filtered;
        filtered.reserve(trans.size());

        for (int item : trans) {
            if (global_freq[item] >= min_count) {
                filtered.push_back(item);
            }
        }

        if (filtered.empty()) continue;

        sort(filtered.begin(), filtered.end(), [&](int a, int b) {
            if (global_freq[a] != global_freq[b]) return global_freq[a] > global_freq[b];
            return a < b;
        });

        tree->add_transaction(filtered, 1);
    }

    return tree;
}

FPTree* build_conditional_tree(
    const vector<pair<vector<int>, int>>& pattern_base,
    int min_count
) {
    vector<int> local_freq(MAX_ITEM, 0);

    for (const auto& entry : pattern_base) {
        const vector<int>& path = entry.first;
        int cnt = entry.second;
        for (int item : path) {
            local_freq[item] += cnt;
        }
    }

    FPTree* tree = new FPTree();

    for (const auto& entry : pattern_base) {
        const vector<int>& path = entry.first;
        int cnt = entry.second;

        vector<int> filtered;
        filtered.reserve(path.size());

        for (int item : path) {
            if (local_freq[item] >= min_count) {
                filtered.push_back(item);
            }
        }

        if (filtered.empty()) continue;

        sort(filtered.begin(), filtered.end(), [&](int a, int b) {
            if (local_freq[a] != local_freq[b]) return local_freq[a] > local_freq[b];
            return a < b;
        });

        tree->add_transaction(filtered, cnt);
    }

    return tree;
}

void save_pattern(vector<pair<vector<int>, int>>& results, vector<int> pattern, int support_count) {
    sort(pattern.begin(), pattern.end());
    results.push_back({std::move(pattern), support_count});
}

void enumerate_single_path(
    FPNode* start,
    const vector<int>& suffix,
    vector<pair<vector<int>, int>>& results
) {
    vector<pair<int, int>> path;
    FPNode* cur = start;

    while (cur) {
        path.push_back({cur->item, cur->count});
        if (cur->children.empty()) break;
        cur = cur->children.begin()->second;
    }

    int m = (int)path.size();

    for (int mask = 1; mask < (1 << m); ++mask) {
        vector<int> pattern = suffix;
        int support_count = INT_MAX;

        for (int i = 0; i < m; ++i) {
            if (mask & (1 << i)) {
                pattern.push_back(path[i].first);
                support_count = min(support_count, path[i].second);
            }
        }

        save_pattern(results, std::move(pattern), support_count);
    }
}

void fp_growth(FPTree* tree, vector<int>& suffix, int min_count,
               vector<pair<vector<int>, int>>& results) {
    if (tree->empty()) return;

    if (tree->has_single_path()) {
        if (!tree->root->children.empty()) {
            enumerate_single_path(tree->root->children.begin()->second, suffix, results);
        }
        return;
    }

    vector<pair<int, int>> items;
    for (int item = 0; item < MAX_ITEM; ++item) {
        if (tree->freq[item] >= min_count) {
            items.push_back({item, tree->freq[item]});
        }
    }

    sort(items.begin(), items.end(), [&](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second < b.second;
        return a.first < b.first;
    });

    for (const auto& item_freq : items) {
        int item = item_freq.first;
        int support_count = item_freq.second;

        vector<int> new_suffix = suffix;
        new_suffix.push_back(item);
        save_pattern(results, new_suffix, support_count);

        vector<pair<vector<int>, int>> pattern_base;
        FPNode* node = tree->header_head[item];

        while (node) {
            int cnt = node->count;
            vector<int> path;

            FPNode* p = node->parent;
            while (p && p->item != -1) {
                path.push_back(p->item);
                p = p->parent;
            }

            reverse(path.begin(), path.end());

            if (!path.empty()) {
                pattern_base.push_back({std::move(path), cnt});
            }

            node = node->next;
        }

        FPTree* cond_tree = build_conditional_tree(pattern_base, min_count);
        if (!cond_tree->empty()) {
            fp_growth(cond_tree, new_suffix, min_count, results);
        }
        delete cond_tree;
    }
}

void write_results(const string& filename,
                   const vector<pair<vector<int>, int>>& results,
                   int total_transactions) {
    ofstream fout(filename);
    if (!fout) {
        throw runtime_error("Cannot open output file: " + filename);
    }

    fout << fixed << setprecision(4);

    for (const auto& entry : results) {
        const vector<int>& pattern = entry.first;
        int count = entry.second;

        for (size_t i = 0; i < pattern.size(); ++i) {
            if (i) fout << ",";
            fout << pattern[i];
        }

        double support = static_cast<double>(count) / total_transactions;
        fout << ":" << support + 1e-12 << "\n";
    }
}

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " [min_support] [input_file] [output_file]\n";
        return 1;
    }

    try {
        double min_support = stod(argv[1]);
        string input_file = argv[2];
        string output_file = argv[3];

        if (min_support < 0.0 || min_support > 1.0) {
            cerr << "min_support must be in [0, 1]\n";
            return 1;
        }

        vector<vector<int>> transactions = read_transactions_fast(input_file);
        int n = (int)transactions.size();

        if (n == 0) {
            ofstream fout(output_file);
            return 0;
        }

        int min_count = (int)ceil(min_support * n - 1e-12);

        FPTree* tree = build_fp_tree(transactions, min_count);

        vector<pair<vector<int>, int>> results;
        vector<int> suffix;
        fp_growth(tree, suffix, min_count, results);

        delete tree;

        sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });

        vector<pair<vector<int>, int>> unique_results;
        unique_results.reserve(results.size());

        for (const auto& r : results) {
            if (unique_results.empty() || unique_results.back().first != r.first) {
                unique_results.push_back(r);
            } else {
                unique_results.back().second = max(unique_results.back().second, r.second);
            }
        }

        write_results(output_file, unique_results, n);
    } catch (const exception& e) {
        cerr << e.what() << '\n';
        return 1;
    }

    return 0;
}