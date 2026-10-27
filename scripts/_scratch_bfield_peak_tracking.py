# Scratch notes: future work for fitting peaks vs. magnetic field (Zeeman splitting).
#
# Pulled from PL_bfield_driver.py before that script was deleted (its contour-plot
# functionality is now handled by steady_state_driver.py --mode contour-plot).
# This is NOT runnable code — it predates the current fitting_functions/steady_state
# API and needs to be rebuilt against PLFitter / fit_results.csv (e.g. via
# plot_peak_param, or a new quadratic-fit helper) rather than restored as-is.
#
# Intent: fit N Lorentzian/Gaussian peaks per spectrum across a B-field sweep,
# then fit each peak's energy vs. field to a quadratic to characterize splitting.

# Fit parameters: [amp1, cen1, wid1, amp2, cen2, wid2, ...]
# p0 = [
#     20/260, 1.31, 0.006,
#     90/260, 1.33, 0.006,
#     260/260, 1.34, 0.004,
#     80/260, 1.35, 0.005,
#     20/260, 1.355, 0.005,
# ]

# bounds = [
#     [0, None], [1.307, 1.315], [0.001, 0.1],
#     [0, None], [1.3,   1.33],  [0.001, 0.01],
#     [0, None], [1.32,  1.35],  [0.001, 0.01],
#     [0, None], [1.33,  1.356], [0.001, 0.01],
#     [0, None], [1.345, 1.36],  [0.001, 0.01],
# ]

# # ---- Peak positions vs. B-field (only when do_fit is True) ---- #
# if do_fit and peak_locs:
#     n_peaks = len(p0) // 3
#     by_peak = [[] for _ in range(n_peaks)]
#     for bfield in bfields:
#         locs = peak_locs[str(bfield)]
#         for j in range(n_peaks):
#             by_peak[j].append(locs[j])

#     pquad_init = [-3, 0, 1.4]
#     for i in range(n_peaks):
#         quad_fit = minimize(
#             diffquad, pquad_init, args=(bfields, by_peak[i])
#         ).get('x')
#         a = round(quad_fit[0], 5)
#         b = round(quad_fit[1], 5)
#         c = round(quad_fit[2], 2)
#         label = f"$y_{i+1}$ = {a}$x^2$ + {b}$x$ + {c}"
#         plt.figure(dpi=400)
#         plt.plot(bfields, by_peak[i], "r.")
#         plt.title(f"Peak #{i+1}, {label}")
#         plt.plot(bfields, quadratic(quad_fit, bfields))
#         plt.show()
