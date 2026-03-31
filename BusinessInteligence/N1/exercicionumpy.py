import numpy as np

print("--- Exercício 1 ---")

arr1 = np.arange(10)
print("Original:", arr1)

arr1[5:9] = 0
print("Alterado:", arr1)

print("\n--- Exercício 2 ---")
arr2 = np.arange(6).reshape(3, 2)
print("Matriz:\n", arr2)
# Imprima o shape
print("Shape:", arr2.shape)

print("2ª linha:", arr2[1])

print("\n--- Exercício 3 ---")

arr3 = np.arange(6).reshape(3, 2)
print("Matriz:\n", arr3)

print("2ª coluna:\n", arr3[:, 1])

print("\n--- Exercício 4 ---")

arr4 = np.arange(20).reshape(4, 5)
print("Matriz:\n", arr4)

print("Elementos da 3ª linha:", arr4[2])

print("\n--- Exercício 5 ---")

arr5 = np.arange(20).reshape(4, 5)
print("Matriz:\n", arr5)

print("Elementos (linhas 1-2, colunas 2-3):\n", arr5[0:2, 1:3])

print("\n--- Exercício 6 ---")

arr6 = np.arange(20).reshape(4, 5)
print("Matriz:\n", arr6)

print("Elementos (linhas 2-3, colunas 1-3):\n", arr6[1:3, 0:3])