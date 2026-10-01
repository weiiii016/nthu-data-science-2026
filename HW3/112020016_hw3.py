#  HW3 - CMA-ES Optimizer  (v3: warmup + stagnation restart)

import numpy as np
from HomeworkFramework import Function


class RS_optimizer(Function):
    def __init__(self, target_func):
        super().__init__(target_func)
        self.lower  = self.f.lower(target_func)
        self.upper  = self.f.upper(target_func)
        self.dim    = self.f.dimension(target_func)
        self.target_func      = target_func
        self.eval_times       = 0
        self.optimal_value    = float("inf")
        self.optimal_solution = np.empty(self.dim)

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    # ── 共用 evaluate wrapper ──────────────────────────────
    def _eval(self, x):
        val = self.f.evaluate(self.target_func, x)
        self.eval_times += 1
        if val == "ReachFunctionLimit":
            return None
        val = float(val)
        if val < self.optimal_value:
            self.optimal_value       = val
            self.optimal_solution[:] = x
        return val

    # ── 單次 CMA-ES ────────────────────────────────────────
    # 停止條件：
    #   (A) FES 耗盡 → return False
    #   (B) 連續 stagnation_limit 代無改善 → 觸發 restart
    # sigma 自然縮小到 1e-12 = 正常精確收斂，不觸發 restart
    def _cmaes(self, FES, init_mean, init_sigma, stagnation_limit=30):
        n          = self.dim
        mean       = init_mean.copy()
        sigma      = init_sigma
        range_size = self.upper - self.lower

        lam     = 4 + int(3 * np.log(n))
        mu      = lam // 2
        raw_w   = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = raw_w / raw_w.sum()
        mueff   = 1.0 / (weights ** 2).sum()

        cc    = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs    = (mueff + 2) / (n + mueff + 5)
        c1    = 2 / ((n + 1.3) ** 2 + mueff)
        cmu   = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n+2)**2 + mueff))
        damps = 1 + 2 * max(0.0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN  = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))

        pc = np.zeros(n); ps = np.zeros(n)
        B  = np.eye(n);   D  = np.ones(n)
        C  = np.eye(n);   invsqrtC = np.eye(n)
        eigeneval   = 0
        no_improve  = 0
        best_so_far = self.optimal_value

        while self.eval_times < FES:
            if no_improve >= stagnation_limit:
                return True

            remaining = FES - self.eval_times
            cur_lam   = min(lam, remaining)

            arz = np.random.randn(cur_lam, n)
            arx = mean + sigma * (arz @ (B * D).T)
            arx = np.clip(arx, self.lower, self.upper)

            fitness = []
            for i in range(cur_lam):
                if self.eval_times >= FES:
                    return False
                val = self._eval(arx[i])
                if val is None:
                    return False
                fitness.append((val, i))

            if not fitness:
                break

            # 更新停滯計數
            if self.optimal_value < best_so_far:
                best_so_far = self.optimal_value
                no_improve  = 0
            else:
                no_improve += 1

            fitness.sort(key=lambda x: x[0])
            sel_count = min(mu, len(fitness))
            idx       = [f[1] for f in fitness[:sel_count]]
            w_used    = weights[:sel_count] / weights[:sel_count].sum()

            old_mean = mean.copy()
            mean     = w_used @ arx[idx]

            ps = ((1 - cs) * ps
                  + np.sqrt(cs * (2-cs) * mueff)
                  * invsqrtC @ (mean - old_mean) / sigma)

            hs = (np.linalg.norm(ps)
                  / np.sqrt(1 - (1-cs)**(2 * self.eval_times / lam))
                  / chiN) < (1.4 + 2/(n+1))

            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma  = np.clip(sigma, 1e-12, range_size)   # 允許縮到 1e-12

            pc = ((1 - cc) * pc
                  + hs * np.sqrt(cc*(2-cc)*mueff)
                  * (mean - old_mean) / sigma)

            artmp = (arx[idx] - old_mean) / sigma
            C = ((1 - c1 - cmu) * C
                 + c1 * (np.outer(pc, pc) + (1-hs)*cc*(2-cc)*C)
                 + cmu * (w_used * artmp.T) @ artmp)

            if self.eval_times - eigeneval > lam / (c1+cmu) / n / 10:
                eigeneval = self.eval_times
                C = np.triu(C) + np.triu(C, 1).T
                eigval, B = np.linalg.eigh(C)
                D         = np.sqrt(np.maximum(eigval, 1e-20))
                invsqrtC  = B @ np.diag(1.0/D) @ B.T

            #print(f"  FE={self.eval_times:4d} | best={self.optimal_value:.4e}"
            #      f" | σ={sigma:.2e} | stag={no_improve}")

        return False

    # ── 主流程 ─────────────────────────────────────────────
    def run(self, FES):
        n          = self.dim
        range_size = self.upper - self.lower

        # Phase 1：隨機熱身（前 2% FES，最多 50 個點）
        warmup = max(5, min(int(FES * 0.02), 50))
        print(f"[Warmup] {warmup} random evals")
        for _ in range(warmup):
            if self.eval_times >= FES:
                return
            val = self._eval(np.random.uniform(self.lower, self.upper, n))
            if val is None:
                return

        # Phase 2：CMA-ES，停滯就 restart（sigma 塌陷不 restart）
        init_sigma = range_size / 4.0
        restart    = 0

        while self.eval_times < FES:
            noise     = 0.0 if restart == 0 else range_size * 0.1 * (0.5**restart)
            init_mean = np.clip(
                self.optimal_solution + np.random.uniform(-noise, noise, n),
                self.lower, self.upper
            )
            # restart 時 sigma 縮半，讓搜索集中在已知好區域附近
            cur_sigma = max(init_sigma / (2**restart), range_size * 1e-3)

            print(f"\n[Restart #{restart}] FE={self.eval_times} | σ={cur_sigma:.3e}")
            stagnated = self._cmaes(FES, init_mean, cur_sigma)
            restart  += 1

            if not stagnated:   # FES 耗盡
                break


# ==============================================================
if __name__ == '__main__':
    func_num = 1
    fes_map  = {1: 1000, 2: 1500, 3: 2000, 4: 2500}

    while func_num < 5:
        fes = fes_map[func_num]
        op  = RS_optimizer(func_num)
        op.run(fes)

        best_input, best_value = op.get_optimal()
        print(f"\n[Function {func_num}] best = {best_value}")

        student_id = __file__.split('_')[0]
        out_path   = f"{student_id}_function{func_num}.txt"
        with open(out_path, 'w+') as f:
            for v in best_input:
                f.write(f"{v}\n")
            f.write(f"{best_value}\n")

        print(f"Saved → {out_path}\n{'='*50}")
        func_num += 1
