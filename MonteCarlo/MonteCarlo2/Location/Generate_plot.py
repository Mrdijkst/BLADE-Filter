" Evaluation of the results of the experiments Monte Carlo"

import numpy as np
import matplotlib.pyplot as plt



inu = 10
LossMat_1 = np.load('/Users/MathijsDijkstra/University/Bachelors/Third year Econometrics/Robust-QLE-Model/MonteCarlo/MonteCarlo2/Volatility_FINAL/eps1/mLoss_eps1_dNu10_Eps1.0.npy')
LossMat_99 = np.load('/Users/MathijsDijkstra/University/Bachelors/Third year Econometrics/Robust-QLE-Model/MonteCarlo/MonteCarlo2/Volatility_FINAL/eps0.99/mLoss_eps0.99_dNu10_Eps0.99.npy')
LossMat_98 = np.load('/Users/MathijsDijkstra/University/Bachelors/Third year Econometrics/Robust-QLE-Model/MonteCarlo/MonteCarlo2/Volatility_FINAL/eps0.98/mLoss_eps0.98_dNu10_Eps0.98.npy')
LossMat_97 = np.load('/Users/MathijsDijkstra/University/Bachelors/Third year Econometrics/Robust-QLE-Model/MonteCarlo/MonteCarlo2/Volatility_FINAL/eps0.97/mLoss_eps0.97_dNu10_Eps0.97.npy')


iLenGamma1Grid_eps1 = 200
iLenGamma1Grid_eps99 = 102
iLenGamma1Grid_eps98 = 100
iLenGamma1Grid_eps97 = 100


MeanLoss_eps1 = np.nanmean(LossMat_1, axis=2)
MeanLoss_eps99 = np.nanmean(LossMat_99, axis=2)
MeanLoss_eps98 = np.nanmean(LossMat_98, axis=2)
MeanLoss_eps97 = np.nanmean(LossMat_97, axis=2)

gamma1_grid_eps1 = np.linspace(-4, 2.5, iLenGamma1Grid_eps1)
gamma1_grid_eps99 = np.linspace(-4, 2.1, iLenGamma1Grid_eps99)
gamma1_grid_eps98 = np.linspace(-4, 2.0, iLenGamma1Grid_eps98)
gamma1_grid_eps97 = np.linspace(-4, 2.0, iLenGamma1Grid_eps97)


interested_row = 2

gamma1_star_index = np.argmin(MeanLoss_eps1[interested_row, :-1])
gamma1_star = gamma1_grid_eps1[gamma1_star_index]



# plt.figure(figsize=(8,5))

# MAE curve
if interested_row == 0:
    type = "MAE"
if interested_row == 1:
    type = "Barron 1.5"
if interested_row == 2:
    type = "MSE"
if interested_row == 3:
    type = "Barron 4"
if interested_row == 4:
    type = "Loglik"
# plt.plot(gamma1_grid_eps1, MeanLoss_eps1[intrested_row, :-1], label=f"{type}")

# # Oracle horizontal line
# plt.axhline(MeanLoss_eps1[intrested_row, -1],
#             linestyle="--",
#             color="black",
#             label="Oracle")

# # Vertical line at gamma*
# plt.axvline(gamma1_star,
#             linestyle=":",
#             color="red",
#             label=r"$\gamma_1^*$")

# # Optional: marker at minimum
# plt.scatter(gamma1_star,
#             MeanLoss_eps1[intrested_row, gamma1_star_index],
#             color="red",
#             zorder=5)

# plt.xlabel(r"$\gamma_1$")
# plt.ylabel("Average MAE")
# plt.legend()
# plt.show()

## Now we want to compare the curves across different contamination levels
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12
})

loss_name = type

plt.figure(figsize=(9,6))

eps_data = [
    ("1.00", gamma1_grid_eps1, MeanLoss_eps1),
    ("0.99", gamma1_grid_eps99, MeanLoss_eps99),
    ("0.98", gamma1_grid_eps98, MeanLoss_eps98),
    ("0.97", gamma1_grid_eps97, MeanLoss_eps97),
]

colors = ["black", "blue", "green", "red"]

gamma_star_list = []

for (eps_label, gamma_grid, MeanLoss), color in zip(eps_data, colors):

    mse_curve = MeanLoss[interested_row, :-1]
    eps = round(1-float(eps_label),3)
    plt.plot(gamma_grid,
             mse_curve,
             linewidth=1.5,
             color=color,
             label=rf"$\varepsilon={eps}$")

    gamma_index = np.nanargmin(mse_curve)
    gamma_star = gamma_grid[gamma_index]
    gamma_star_list.append(gamma_star)

    plt.scatter(gamma_star,
                mse_curve[gamma_index],
                color=color,
                s=50,
                zorder=5)
    oracle_value = MeanLoss[interested_row, -1]


# Remove grid (default is off, but ensure)
plt.grid(False)
plt.xlim(-2.5, 2.5)
plt.ylim(0,0.04)



# Add optimal gamma values as vertical dashed lines with annotation
for gamma_star, color in zip(gamma_star_list, colors):
    plt.axvline(gamma_star,
                linestyle=":",
                color=color,
                alpha=0.6)
    # Get axis limits
ymin, ymax = plt.ylim()

ax = plt.gca()

for gamma_star in gamma_star_list:

    # vertical dotted line
    ax.axvline(gamma_star,
               linestyle=":",
               color="black",
               alpha=0.5)

    # small tick on axis
    ax.plot([gamma_star, gamma_star],
        [0, -0.015],                 # go downward instead of upward
        transform=ax.get_xaxis_transform(),
        color="black",
        linewidth=1,
        clip_on=False)


    # vertical number BELOW the x-axis
    ax.text(gamma_star,
            -0.03,                       # negative = below axis
            rf"${gamma_star:.2f}$",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha='center',
            va='top',
            color="black",
            fontsize=11)


plt.xlabel(r"$\gamma_1$", fontsize=13)
plt.ylabel(r"Average " + loss_name, fontsize=13)
plt.legend(frameon=False)
plt.tight_layout()

# Save as vector PDF (LaTeX ready)s
# save to '/Users/MathijsDijkstra/University/Bachelors/Third year Econometrics/Robust-QLE-Model/MonteCarlo/MonteCarlo2/Volatility_FINAL/Pictures'
plt.savefig("/Users/MathijsDijkstra/University/Bachelors/Third year Econometrics/Robust-QLE-Model/MonteCarlo/MonteCarlo2/Volatility_FINAL/Pictures/MSE_all_eps.pdf", format="pdf")
plt.show()
