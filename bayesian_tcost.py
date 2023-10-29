#############################           Alessandro Canzonieri               ##########################
# free code from https://github.com/babayaga102/A-Bayesian-derivation-of-the-square-root-law-of-market-impact/tree/main
# The code is a bit rough and not well commented
# I have not been able to reproduce figure 4 and it hink there is an error on the calculations
# Note, the references refer to the version v1 of the paper https://arxiv.org/pdf/2303.08867v1.pdf which has a lot details

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf, gamma
from scipy import stats
import seaborn as sns

INFORMED_TRADERS = 0.035  # true nu = fraction of Informed Traders in one experiment equivalent to the number of non-random transactions in the market

sqrt2 = np.sqrt(2)


def p_t(chi):
    """
    eq. (11) - expected price_t
    chi ~ N(nu * sqrt(t), sqrt(1 - nu^2))
    """
    return 0.5 + 0.5 * erf(chi / sqrt2)


def p_t_bar(chi, nu_bar_t):
    """
    eq. (19) - expected price_t, given nu_bar_t = nu_bar * np.sqrt(t)
    chi ~ N(nu * sqrt(t), sqrt(1 - nu^2))
    nu_bar_t = nu * sqrt(t)

    """
    num = erf(chi / sqrt2) + erf((nu_bar_t - chi) / sqrt2)
    denom = erf((nu_bar_t + chi) / sqrt2) + erf((nu_bar_t - chi) / sqrt2)
    return num / denom


def sqrtfit(x, B):
    return B * np.sqrt(x)


def powerfit(x, B, d):
    return B * (x ** d)

##################################################################################################################
# Header: monte carlo simulation for fixed nu (=INFORMED_TRADERS) and across different Q
n_samples = 10000

nu = INFORMED_TRADERS
# nu = 0.1

all_qtys = np.arange(1, 100)
market_impact = []  # market impact, p_T - p_0
std_t = []
for q in all_qtys:
    t_lim = int(q / nu)  # map from q to duration t; TODO: Idea -> map from (q,t) to calibrate the INFORMED_TRADERS

    chi_0 = np.random.normal(nu * np.sqrt(1), np.sqrt(1 - nu ** 2), n_samples)
    chi_t = np.random.normal(nu * np.sqrt(t_lim), np.sqrt(1 - nu ** 2), n_samples)

    simulated_p0 = p_t(chi_0)
    simulated_pt = p_t(chi_t)
    market_impact.append((simulated_pt - simulated_p0).mean())

    avg_p_t = simulated_pt.mean()  # average price
    stddev = np.sqrt(((simulated_pt - avg_p_t) ** 2).mean())
    std_t.append(stddev)

# fit sqrt-law
t_fit = min(int(1 / nu), len(all_qtys))
[B10], pcov1 = curve_fit(sqrtfit, all_qtys[:t_fit], market_impact[:t_fit], p0=[1])
[B10_std] = np.sqrt(pcov1.diagonal())
chisq_sqrtlaw = ((market_impact[:t_fit] - sqrtfit(all_qtys[:t_fit], B10)) ** 2).sum()
print(f"fitting B * sqrt(Q)\nB +/- 1 std= {B10:.6f} +/- {B10_std:.6f}\n{chisq_sqrtlaw = :.6f}")

plt.close('all')
plt.figure(1, figsize=(9, 6))  # Fig. 1, page 7, left plot
# plt.xscale('log')
# plt.yscale('log')
plt.title(f"nu = {nu}")
plt.ylabel("$p_t - p_1$")
plt.xlabel("Q")
plt.errorbar(all_qtys, market_impact, color="blue", fmt=".")  # market impact
plt.errorbar(all_qtys, std_t, color="red", fmt=".")  # standard deviation of p_t
plt.plot(all_qtys, sqrtfit(all_qtys, B10), color="black", linestyle='--')  # fitted market impact via sqrt(Q)
plt.axvline(1 / INFORMED_TRADERS, color="green", linestyle='--')  # at 1 / nu the sqrt-law stops working
plt.show()

if False:
    import seaborn as sns
    _, ax = plt.subplots(1, 1, figsize=(9, 6))
    sns.histplot(simulated_pt - simulated_p0, bins=100, ax=ax, stat='percent')
    ax.axvline((simulated_pt - simulated_p0).mean(), color='red')

##################################################################################################################
# Header: Fig. 1, page 7; left plot, monte carlo simulation for fixed Q and across different nu

Q = 10

std_t_Q = []
market_impact_Q = []
all_nus = np.logspace(-2.3, 0, 25)
for nu in all_nus:
    t_max = Q / nu
    chi_0_Q = np.random.normal(all_nus[0] * np.sqrt(1), np.sqrt(1 - all_nus[0] ** 2), 100000)
    chi_t_Q = np.random.normal(nu * np.sqrt(t_max), np.sqrt(1 - nu ** 2), 100000)

    simulated_p0_Q = p_t(chi_0_Q)
    simulated_pt_Q = p_t(chi_t_Q)
    market_impact_Q.append((simulated_pt_Q - simulated_p0_Q).mean())

    p_t_Q_avg = simulated_pt_Q.mean()
    std_t_Q.append(np.sqrt(((( (simulated_pt_Q - p_t_Q_avg) ** 2).sum()) / 100000)))

# fit sqrt-law
n_fit = max(3, np.where(all_nus <= 1 / Q, 1, 0).sum())  # at nu = 1 / Q the sqrt-law stops working, exclude from fit
[B20], pcov2 = curve_fit(sqrtfit, all_nus[:n_fit], market_impact_Q[:n_fit], p0=[1])
[B20_std] = np.sqrt(pcov2.diagonal())
chisq_sqrtlaw = ((market_impact_Q[:n_fit] - sqrtfit(all_nus[:n_fit], B20)) ** 2).sum()
print(f"fit B * sqrt(nu)\nB +/- 1 std= {B20:.6f} +/- {B20_std:.6f}\n{chisq_sqrtlaw = :.6f}")

plt.figure(2, figsize=(9, 6))  # Fig. 1, page 7
plt.title(f"Q = {Q}")
plt.ylabel("$p_t - p_1$")
plt.xlabel("nu")
# plt.yscale('log')
# plt.xscale('log')
plt.errorbar(all_nus, market_impact_Q, color="blue", fmt=".")
plt.errorbar(all_nus, std_t_Q, color="red", fmt=".")
plt.plot(all_nus, sqrtfit(all_nus, B20), color="black", linestyle='--')
plt.axvline(1 / Q, color="green", linestyle='--')  # at nu = 1 / Q the sqrt-law stops working
plt.show()

##################################################################################################################
# Header: Fig. 2, page 9; for t=1..10000, sample the price impact n_samples times, and then take the average

nu = 0.01
nu_bar = 0.03
n_samples = 5000
all_t = np.arange(1, 10001)
expected_pt_bar = []
for t in all_t:
    chi_0_bar = np.random.normal(nu * np.sqrt(1), np.sqrt(1 - nu ** 2), n_samples)
    chi_t_bar = np.random.normal(nu * np.sqrt(t), np.sqrt(1 - nu ** 2), n_samples)
    nu_bar_t = nu_bar * np.sqrt(t)
    avg_impact = np.mean(p_t_bar(chi_t_bar, nu_bar_t) - p_t_bar(chi_0_bar, nu_bar))
    expected_pt_bar.append(avg_impact)

# fit small t -> linear with fitted power d40 being close to 1
t_trans = int(1 / (nu_bar ** 2))
[B30, d30], pcov3 = curve_fit(powerfit, all_t[:t_trans], expected_pt_bar[:t_trans], p0=[1, 1])
B30_std, d30_std = np.sqrt(pcov3.diagonal())
chisq_nu_bar_A = ((expected_pt_bar[:t_trans] - powerfit(all_t[:t_trans], B30, d30)) ** 2).sum()

# fit large t -> sqrt with fitted power d40 being close to 0.5
[B40, d40], pcov4 = curve_fit(powerfit, all_t[t_trans:], expected_pt_bar[t_trans:], p0=[1, 1])
B40_std, d40_std = np.sqrt(pcov4.diagonal())
chisq_nu_bar_B = ((expected_pt_bar[t_trans:] - powerfit(all_t[t_trans:], B40, d40)) ** 2).sum()

print("Linear part: fit B30 * x^d30\tSqrt part: fit B40 * x^d40")
print(f"B30 +/- 1 std  = {B30:10.6f} +/- {B30_std:.6f}\nd30 +/- 1 std  = {d30:10.6f} +/- {d30_std:.6f}\n{chisq_nu_bar_A = :10.6f}\n")
print(f"B40 +/- 1 std  = {B40:10.6f} +/- {B40_std:.6f}\nd40 +/- 1 std  = {d40:10.6f} +/- {d40_std:.6f}\n{chisq_nu_bar_B = :10.6f}\n")

plt.figure(3, figsize=(9, 6))  # Fig. 2, page 9
plt.title(f"nu={nu}, nu_bar = {nu_bar}")
plt.ylabel("$E[p_t - p_1]$")
plt.xlabel("t")
plt.yscale('log')
plt.xscale('log')
plt.plot(all_t, expected_pt_bar, "b-")
plt.plot(all_t, powerfit(all_t, B30, d30), color="black", linestyle='--', linewidth=2)  # linear
plt.plot(all_t, powerfit(all_t, B40, d40), color="magenta", linestyle='--', linewidth=2)  # sqrt
plt.axvline(1 / (nu_bar ** 2), color="grey", linestyle='--')
plt.axvline(1 / (nu ** 2), color="grey", linestyle='--')
plt.show()

##################################################################################################################
# Header: Fig. 4, page 11; buy order from t=1..400, then decaying market impact afterwards

t_buy = 400
nu = 0.01
all_t = np.arange(1, 2000)
n_samples = 10000
Q_fixed = INFORMED_TRADERS * t_buy

market_impact_decay = []
for t in all_t:
    if t <= t_buy:  # informed trader is still buying
        q = t * INFORMED_TRADERS
        chi_0_decay_buy = np.random.normal(nu * np.sqrt(1), np.sqrt(1 - nu ** 2), n_samples)
        chi_t_decay_buy = np.random.normal(q / np.sqrt(t), np.sqrt(1 - ((nu * q) / t)), n_samples)
        market_impact_decay.append((p_t(chi_t_decay_buy) - p_t(chi_0_decay_buy)).mean())
    else:
        chi_0_decay_buy = np.random.normal(nu * np.sqrt(1), np.sqrt(1 - nu ** 2), n_samples)
        chi_t_decay_buy = np.random.normal(Q_fixed / np.sqrt(t), np.sqrt(1 - ((nu * Q_fixed) / t)), n_samples)
        market_impact_decay.append((p_t(chi_t_decay_buy) - p_t(chi_0_decay_buy)).mean())

(B60, d60), pcov6 = curve_fit(powerfit, all_t[:t_buy], market_impact_decay[:t_buy], p0=[1, 1])
B60_std, d60_std = np.sqrt(pcov6.diagonal())
chisq_decay_1 = ((market_impact_decay[:t_buy] - powerfit(all_t[:t_buy], B60, d60)) ** 2).sum()

(B70, d70), pcov7 = curve_fit(powerfit, all_t[t_buy:], market_impact_decay[t_buy:], p0=[1, 1])
B70_std, d70_std = np.sqrt(pcov7.diagonal())
chisq_decay_2 = ((market_impact_decay[t_buy:] - powerfit(all_t[t_buy:], B70, d70)) ** 2).sum()

print(f"fit B60 * t^d60 upward till t={t_buy}")
print(f"B60 +/- 1 std = {B60:10.6f} +/- {B60_std:.6f}\nd60 +/- 1 std = {d60:10.6f} +/- {d60_std:.6f}\n{chisq_decay_1 = :10.6f}")
print(f"\nfit B70 * t^d70 decay downward after t={t_buy}")
print(f"B70 +/- 1 std = {B70:10.6f} +/- {B70_std:.6f}\nd70 +/- 1 std = {d70:10.6f} +/- {d70_std:.6f}\n{chisq_decay_2 = :10.6f}")

plt.figure(4, figsize=(9, 6))  # Fig. 4, page 11
plt.title("decay")
plt.ylabel("$E[P_t - p_1]$")
plt.xlabel("t")
plt.ylim(0, max(market_impact_decay) * 1.2)
plt.plot(all_t, market_impact_decay, color="blue", linestyle="-")
plt.plot(all_t, powerfit(all_t, B60, d60), color="green", linestyle='dashed')
plt.plot(all_t, powerfit(all_t, B70, d70), color="red", linestyle='dashed')
plt.show()


##################################################################################################################
# Header: code was not verified below this line
# k-impact

def cumfunction(a, b, mean, std):
    return stats.norm.cdf(b, mean, std) - stats.norm.cdf(a, mean, std)


def p_t_k_2(chi, k):
    return 0.5 + (chi / sqrt2) * ((gamma((1 + k) / 2)) / (gamma(k / 2))) * ((cumfunction((1 - (k / 2)), 3 / 2, 0, 1)) / (cumfunction(((1 - k) / 2), 1 / 2, 0, 1)))


k = [0.02, 0.05, 0.1, 0.2, 0.5, 1]
all_t = np.logspace(0, 3.5, 250)

delta_p_t_k_0 = []
delta_p_t_k_1 = []
delta_p_t_k_2 = []
delta_p_t_k_3 = []
delta_p_t_k_4 = []
delta_p_t_k_5 = []

for t in all_t:
    nu = 0.1

    chi0_k = np.random.normal(nu * np.sqrt(1), np.sqrt(1 - nu ** 2), 100000)
    chi_k_t = np.random.normal(nu * np.sqrt(t), np.sqrt(1 - nu ** 2), 100000)

    p_t_k_0_avg_0 = p_t_k_2(chi0_k, k[0]).sum() / 100000
    p_t_k_1_avg_0 = p_t_k_2(chi0_k, k[1]).sum() / 100000
    p_t_k_2_avg_0 = p_t_k_2(chi0_k, k[2]).sum() / 100000
    p_t_k_3_avg_0 = p_t_k_2(chi0_k, k[3]).sum() / 100000
    p_t_k_4_avg_0 = p_t_k_2(chi0_k, k[4]).sum() / 100000
    p_t_k_5_avg_0 = p_t_k_2(chi0_k, k[5]).sum() / 100000

    p_t_k_0_avg_t = p_t_k_2(chi_k_t, k[0]).sum() / 100000
    p_t_k_1_avg_t = p_t_k_2(chi_k_t, k[1]).sum() / 100000
    p_t_k_2_avg_t = p_t_k_2(chi_k_t, k[2]).sum() / 100000
    p_t_k_3_avg_t = p_t_k_2(chi_k_t, k[3]).sum() / 100000
    p_t_k_4_avg_t = p_t_k_2(chi_k_t, k[4]).sum() / 100000
    p_t_k_5_avg_t = p_t_k_2(chi_k_t, k[5]).sum() / 100000

    delta_p_t_k_0.append(p_t_k_0_avg_t - p_t_k_0_avg_0)
    delta_p_t_k_1.append(p_t_k_1_avg_t - p_t_k_1_avg_0)
    delta_p_t_k_2.append(p_t_k_2_avg_t - p_t_k_2_avg_0)
    delta_p_t_k_3.append(p_t_k_3_avg_t - p_t_k_3_avg_0)
    delta_p_t_k_4.append(p_t_k_4_avg_t - p_t_k_4_avg_0)
    delta_p_t_k_5.append(p_t_k_5_avg_t - p_t_k_5_avg_0)

plt.figure(5)
plt.xscale('log')
plt.yscale('log')
plt.title("price impact for different k  [nu = 0.1]")
plt.ylabel("E[P_t -p_1] for different k")
plt.xlabel("t")
plt.ylim(0.0001, 10)
plt.xlim(0.8, 7000)
plt.errorbar(all_t, delta_p_t_k_0, color="blue", fmt="-")
plt.errorbar(all_t, delta_p_t_k_1, color="red", fmt="-")
plt.errorbar(all_t, delta_p_t_k_2, color="salmon", fmt="-")
plt.errorbar(all_t, delta_p_t_k_3, color="gray", fmt="-")
plt.errorbar(all_t, delta_p_t_k_4, color="black", fmt="-")
plt.errorbar(all_t, delta_p_t_k_5, color="orange", fmt="-")
plt.show()