using Flux
using Flux: train!
using Plots
function f(x)
    return (1 - x)*sin(20 * log(x + 0.2))
end

x_training = reshape(range(0, 1, 1000), 1, :)
y_training = f.(x_training)

neural_net = Chain(
    Dense(1 => 64, leakyrelu),
    Dense(64 => 64, leakyrelu),
    Dense(64 => 64, leakyrelu),
    Dense(64 => 1)
)

#loss and opt
loss(m, x, y) = Flux.mse(m(x), y)
opt = Flux.setup(Adam(0.005), neural_net)
# save da best

best_loss = Inf
best_state = Flux.state(neural_net)

# train it
loss_vector = zeros(3000)
for epoch in 1:3000
    train!(loss, neural_net, [(x_training, y_training)], opt)

    current_loss = loss(neural_net, x_training, y_training)
    loss_vector[epoch] = current_loss
    if current_loss < best_loss
        global best_loss = current_loss
        global best_state = Flux.state(neural_net) |> deepcopy
    end
end

x_test = reshape(range(0, 1, 100), 1, :)
y_test = f.(x_test)
Flux.loadmodel!(neural_net, best_state)
inference = neural_net(x_test)

# plot the learning curve
p1 = plot(1:3000, loss_vector,
    xlabel="Epoch", ylabel="MSE Loss",
    title="Learning Curve", label="Loss",
    linewidth=2, yscale=:log10)

# plaot 100 infered vs y_test

p2 = plot(vec(x_test), vec(y_test),
    label="True f(x)", linewidth=2,
    xlabel="x", ylabel="y",
    title="Neural Net vs True Function")
plot!(p2, vec(x_test), vec(inference),
    label="NN Prediction", seriestype=:scatter,
    marker=:cross, markersize=4)

plot(p1, p2, layout=(1, 2), size=(1200, 600),
    margin=5Plots.mm)
savefig("results.png")