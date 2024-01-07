import numpy as np

# Definir a função f(x,y) e o jacobiano Jf(x,y)
def f(x):
    return np.array([x[0] + x[1] - 3, x[0]**2 + x[1]**2 - 9])

def Jf(x):
    return np.array([[1, 1], [2*x[0], 2*x[1]]])

# Definir a aproximação inicial
x0 = np.array([1, 5])

# Definir o critério de parada
tol =  0.0001
max_iter = 20

# Algoritmo de Newton
for i in range(max_iter):
    # Calcular a matriz Jacobiana e a função avaliadas no ponto atual
    J = Jf(x0)
    fx0 = f(x0)

    # Resolver o sistema linear J*delta_x = -f(x0)
    delta_x = np.linalg.solve(J, -fx0)

    # Atualizar o ponto
    x1 = x0 + delta_x

    # Verificar o critério de parada
    if np.linalg.norm(delta_x) < tol:
        print("Solução encontrada em", i+1, "iterações:", x1)
        break

    # Atualizar o ponto inicial
    x0 = x1
else:
    print("O método não convergiu em", max_iter, "iterações.")

# Imprimir o resultado final, mesmo que não tenha convergido
print("Resultado final após", max_iter, "iterações:", x1)
