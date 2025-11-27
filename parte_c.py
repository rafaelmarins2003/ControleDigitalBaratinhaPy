import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from parte_b import ganhos_k

# ==========================================
# 1. DEFINIÇÃO DA PLANTA E CONTROLE
# ==========================================
Ts = 0.01

# Planta G(z)
num_planta = [0, 0.4857, -0.1619]
den_planta = [1, -1.9990, 0.9985]
G_z = ctrl.TransferFunction(num_planta, den_planta, Ts)

Kp, Ki, Kd = ganhos_k(zeta_proj=0.5, wn_proj=3.0, multiplo_z3=37, z4=-0.65, debug=False)

print(f"--- CONFIGURAÇÃO DO CONTROLE ---")
print(f"Ganhos utilizados: Kp={Kp}, Ki={Ki}, Kd={Kd}")

# Montagem do Controlador C(z)
# C(z) = Kp + Ki*z/(z-1) + Kd*(z-1)/z
# Numerador C: (Kp + Ki + Kd)z^2 - (Kp + 2Kd)z + Kd
# Denominador C: z^2 - z
num_ctrl = [(Kp + Ki + Kd), -(Kp + 2*Kd), Kd]
den_ctrl = [1, -1, 0]
C_z = ctrl.TransferFunction(num_ctrl, den_ctrl, Ts)

# ==========================================
# 2. FECHAMENTO DA MALHA
# ==========================================
L_z = C_z * G_z                 # Malha Aberta
T_z = ctrl.feedback(L_z, 1)     # Malha Fechada (Y/R)
T_u = ctrl.feedback(C_z, G_z)   # Esforço de Controle (U/R)

# Verificação rápida de polos
polos = ctrl.poles(T_z)
print(f"Polos de Malha Fechada: {np.abs(polos)}")
if np.any(np.abs(polos) >= 1):
    print(">>> ALERTA: SISTEMA INSTÁVEL! Reduza os ganhos. <<<")
else:
    print(">>> SISTEMA ESTÁVEL. <<<")

# ==========================================
# 3. SIMULAÇÃO E PLOTS
# ==========================================
# Vetor de tempo estendido para garantir captura da acomodação
T_sim = np.arange(0, 6.0, Ts)

plt.figure(figsize=(12, 10))

# --- Gráfico 1: Lugar das Raízes ---
plt.subplot(2, 2, 1)
ctrl.root_locus(L_z, plot=True, grid=True)
plt.plot(np.real(polos), np.imag(polos), 'rX', markersize=10, label='Polos Atuais')
plt.xlim([0.9, 1.1])
plt.ylim([-0.1, 0.1])
plt.title("1. Lugar das Raízes (Root Locus)")
plt.legend()

# --- Gráfico 2: Resposta ao Degrau (Y) ---
plt.subplot(2, 2, 2)
t, y = ctrl.step_response(T_z, T=T_sim)
plt.plot(t, y, linewidth=2)
plt.axhline(1.0, color='r', linestyle='--', label='Ref')
plt.axhline(1.02, color='k', linestyle=':', alpha=0.3) # Faixa 2%
plt.axhline(0.98, color='k', linestyle=':', alpha=0.3)
plt.title("2. Resposta ao Degrau (Posição)")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Métricas
try:
    info = ctrl.step_info(T_z, T=T_sim)
    Mp = info['Overshoot']
    ts = info['SettlingTime']
    plt.plot(info['PeakTime'], info['Peak'], 'ro')
    plt.text(info['PeakTime'], info['Peak']*1.02, f"Mp={Mp:.1f}%")
except:
    Mp, ts = 0, 0

# --- Gráfico 3: Sinal de Controle (U) ---
plt.subplot(2, 1, 2)
t_u, u = ctrl.step_response(T_u, T=T_sim)
plt.plot(t_u, u, 'g', linewidth=2)
plt.title("3. Esforço de Controle (u[k])")
plt.xlabel("Tempo (s)")
plt.ylabel("Sinal de Controle")
plt.grid(True)

# Limites físicos (Exemplo: PWM saturado em 100 ou Tensão 12V)
limite = 100
plt.axhline(limite, color='r', linestyle='--')
plt.axhline(-limite, color='r', linestyle='--')
plt.text(0, limite*1.1, "Saturação Estimada", color='r', fontsize=8)

plt.tight_layout()
plt.show()

# ==========================================
# 4. RELATÓRIO FINAL
# ==========================================
print("\n" + "="*40)
print("VALIDAÇÃO DO PROJETO (PARTE C)")
print("="*40)
print(f"1. Sobressinal (Mp):")
print(f"   - Meta: <= 30%")
print(f"   - Obtido: {Mp:.2f}%")
print(f"   - Status: {'APROVADO' if Mp <= 30 else 'REPROVADO'}")

print(f"\n2. Tempo de Acomodação (ts 2%):")
print(f"   - Meta: < 3.0s")
print(f"   - Obtido: {ts:.4f}s")
print(f"   - Status: {'APROVADO' if ts < 3.0 else 'REPROVADO'}")

u_max = np.max(np.abs(u))
print(f"\n3. Esforço de Controle:")
print(f"   - Pico Máximo: {u_max:.2f}")
print("   - Análise: O pico inicial é natural da ação derivativa.")
print("     Se exceder o limite físico (saturação), o robô real")
print("     será mais lento que a simulação (efeito windup).")