def quickSort(arr, l, r):
    if l < r:
        l = 0
        r = len(arr) - 1

        pi = partition(arr, l, r)

        quickSort(arr, l, pi-1)
        quickSort(arr,pi+1, r)

def partition(arr, l, r):

    pivot = arr[r]

    i = l-1 # 指向比pivot小的元素的最后位置

    for j in range(l, r):

        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i+1], arr[r] = arr[r], arr[i+1]

    return i + 1





