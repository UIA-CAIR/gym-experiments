


def state_to_file(arr):

    arr = arr[0]
    layers = arr.shape[0]

    for l in range(layers):
        rows = arr[l]

        data = ""

        for x in range(rows.shape[0]):
            cols = rows[x]
            for y in range(cols.shape[0]):

                data += str(cols[y]) + " "

            data += "\n"

        print(data)