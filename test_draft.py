irk = IRK(...)

num_sim = 100

xcurrent = ..
for ...
    xcurrent = irk.solve(xcurrent)
    x_sim[i] = xcurrent


# plot
