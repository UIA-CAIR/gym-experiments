import random



memory = Memory(10000000)
action_size = 13

X_0 = []
X_1 = []
Y = []




train_percent = .60

for item in load_all_memories(4):
    s, a, r, s1, t = item
    a_arr = np.zeros(shape=(action_size, ))
    a_arr[a] = 1
    s = np.reshape(s, tuple(reversed(s.shape[1:])))
    s1 = np.reshape(s1, tuple(reversed(s1.shape[1:])))

    X_1.append(s)
    X_0.append(a_arr)
    Y.append(s1)


n_train = int(len(X_0) * train_percent)
n_test = len(X_0) - n_train

X_train = [np.array(X_0[:n_train]), np.array(X_1[:n_train])]
Y_train = np.array(Y[:n_train])

X_test = [np.array(X_0[n_train:n_train+n_test]), np.array(X_1[n_train:n_train+n_test])]
Y_test = np.array(Y[n_train:n_train+n_test])

for i in range(10000):
    model.fit(X_train, Y_train, epochs=1, validation_data=(X_test, Y_test), callbacks=[plot_losses], verbose=1)

    n = 8
    indexes = random.sample(range(n_test), n)

    X = [np.array([X_test[0][_] for _ in indexes]), np.array([X_test[1][_] for _ in indexes])]
    X_pred = model.predict_on_batch(X)

    real = X[1]
    fake = X_pred[:n]

    print(real.shape, fake.shape)

    real_fake = []
    for j in range(len(real)):
        real_fake.append(real[j])
        real_fake.append(fake[j])

    stitch_generated_image(real_fake, "yes")



