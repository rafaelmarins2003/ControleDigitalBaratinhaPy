import control as ctrl
import numpy as np


Ts = 0.01
# --------------------------------------------
# 1. Definição da planta contínua G(s)
# --------------------------------------------

# Numerador e denominador conforme o enunciado
num_s = [32.4, 3240]
den_s = [1, 0.15, -4.9896]
# Observação:
# (s - 2.16)(s + 2.31) = s^2 + (2.31 - 2.16)s - (2.16*2.31)
#                     = s^2 + 0.15s - 4.9896

G_s = ctrl.TransferFunction(num_s, den_s)

# --------------------------------------------
# 2. Discretização manual: usar G(z) (Obitdo a partir dos calculos)
# --------------------------------------------

# G(z) = (0.48571 z − 0.16194) / (z^2 − 1.9989997 z + 0.9985011)
num_z = [0.48571, -0.16194]
den_z = [1, -1.9989997, 0.9985011]

G_z = ctrl.TransferFunction(num_z, den_z, Ts)

# --------------------------------------------
# 3. Polos
# --------------------------------------------

poles_s = ctrl.poles(G_s)
poles_z = ctrl.poles(G_z)

print("Polos de G(s):")
print(poles_s)

print("\nPolos de G(z) (discretizada):")
print(poles_z)

# O sistema não e estavel em malha aberta.
#
# Justificativa no Plano S (Continuo): Existe um polo em +2.16. Para garantir estabilidade,
# todos os polos precisam ter parte real negativa (lado esquerdo do grafico).
# Como temos um polo positivo, a resposta do sistema tende ao infinito.
#
# Justificativa no Plano Z (Digital): Esse mesmo polo instavel se transformou em 1.02.
# No dominio digital, para ser estavel, o valor do polo deve ser menor que 1 (estar dentro do circulo unitario).
# Como 1.02 e maior que 1, confirma-se que o sistema digital tambem e instavel.

# --------------------------------------------
# 4. Verificar estabilidade aberta
# --------------------------------------------

def check_stability_s(poles):
    unstable = [p for p in poles if np.real(p) > 0]
    return len(unstable) == 0

def check_stability_z(poles):
    unstable = [p for p in poles if abs(p) >= 1]
    return len(unstable) == 0

print("\nEstável em malha aberta (s)?", check_stability_s(poles_s))
print("Estável em malha aberta (z)?", check_stability_z(poles_z))

# --------------------------------------------
# 5. Ganho DC
# --------------------------------------------

dc_gain = ctrl.dcgain(G_z)
print("\nGanho DC de G(z):", dc_gain)

# Confirmação de que está de fato correto:

dc_gain_auto = ctrl.dcgain(ctrl.c2d(G_s, Ts, method='zoh'))
print("\nConfirmação do DC de G(z) utilizando função interna:", dc_gain_auto)

