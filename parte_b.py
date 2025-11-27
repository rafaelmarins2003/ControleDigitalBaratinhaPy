import numpy as np
import control as ctrl
import sympy as sp
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURAÇÕES INICIAIS
# ==========================================
Ts = 0.01

# Planta obtida na Parte A
# G(z) = (0.4857z - 0.1619) / (z^2 - 1.9990z + 0.9985)
num_planta = [0, 0.4857, -0.1619]
den_planta = [1, -1.9990, 0.9985]


def ganhos_k(zeta_proj=0.5, wn_proj=3.0, multiplo_z3=8, z4=0.2, debug=False):
    if debug:
        print("\n" + "=" * 50)
        print("   PARTE B - PROJETO ALOCAÇÃO DE POLOS   ")
        print("=" * 50)

    # -----------------------------------------------------
    # 1. ANÁLISE DE REQUISITOS
    # -----------------------------------------------------
    if debug:
        print("\n[1] REQUISITOS DE DESEMPENHO")
        print("-" * 30)

    Mp_target = 0.30  # 30%
    ts_target = 3.0  # 3 segundos

    # Fórmulas de aproximação de 2ª ordem
    ln_mp = np.log(Mp_target)
    zeta_min = np.abs(ln_mp) / np.sqrt((np.pi ** 2 + ln_mp ** 2))
    wn_min = 4 / (zeta_proj * ts_target)

    if debug:
        print(f"Especificações:")
        print(f"  - Sobressinal (Mp) <= {Mp_target * 100}%")
        print(f"  - Tempo de Acomodação (ts) < {ts_target}s")
        print(f"\nRestrições Calculadas:")
        print(f"  - Amortecimento (zeta) >= {zeta_min:.4f}")
        print(f"\nParâmetros Escolhidos para o Projeto:")
        print(f"  -> zeta = {zeta_proj}")
        print(f"  Dado o zeta escolhido, o wn minimo fica wn_min = {wn_min:.4f}")
        print(f"  -> wn   = {wn_proj} rad/s")

    # -----------------------------------------------------
    # 2. ESCOLHA DOS POLOS
    # -----------------------------------------------------
    if debug:
        print("\n[2] ALOCAÇÃO DE POLOS")
        print("-" * 30)

    # Polos Dominantes (s-plane)
    sigma = zeta_proj * wn_proj
    wd = wn_proj * np.sqrt(1 - zeta_proj ** 2)
    s1 = -sigma + 1j * wd
    s2 = -sigma - 1j * wd

    # Mapeamento para z-plane (z = e^sT)
    z1 = np.exp(s1 * Ts)
    z2 = np.exp(s2 * Ts)

    aux = sigma * multiplo_z3
    z3 = np.exp(-Ts * aux)

    if debug:
        print(f"Polos Dominantes (z): {z1:.4f} e {z2:.4f}")
        print(f"Polo Auxiliar (z):    {z3:.4f}")
        print(f"Polo na Origem (z):   {z4}")

    # Polinômio Desejado P_des(z)
    polinomio_desejado = np.poly([z1, z2, z3, z4])
    r_des = np.real(polinomio_desejado)

    if debug:
        print("\nCoeficientes do Polinômio Desejado D(z):")
        print(f"z^4 + ({r_des[1]:.4f})z^3 + ({r_des[2]:.4f})z^2 + ({r_des[3]:.4f})z + ({r_des[4]:.4f})")

    # -----------------------------------------------------
    # 3. CÁLCULO DOS GANHOS (SISTEMA LINEAR)
    # -----------------------------------------------------
    if debug:
        print("\n[3] ÁLGEBRA E CÁLCULO DOS GANHOS")
        print("-" * 30)

    # Variáveis Simbólicas
    z_sym = sp.symbols('z')
    Kp, Ki, Kd = sp.symbols('Kp Ki Kd')

    # Planta Simbólica
    Num_G = num_planta[1] * z_sym + num_planta[2]
    Den_G = den_planta[0] * z_sym ** 2 + den_planta[1] * z_sym + den_planta[2]

    # Controlador PID Simbólico (Posicional)
    Num_C = (Kp + Ki + Kd) * z_sym ** 2 - (Kp + 2 * Kd) * z_sym + Kd
    Den_C = z_sym ** 2 - z_sym

    # Equação Característica: 1 + C(z)G(z) = 0
    Eq_Char = sp.expand(Den_G * Den_C + Num_G * Num_C)

    # Coletar coeficientes
    coeffs_poly = sp.Poly(Eq_Char, z_sym).coeffs()

    if debug:
        print("Equações Algébricas obtidas (Lado Esquerdo):")
        print(f"Coef z^3: {coeffs_poly[1]}")
        print(f"Coef z^2: {coeffs_poly[2]}")
        print(f"Coef z^1: {coeffs_poly[3]}")

    # Montar Sistema Ax = b
    def extrair_coeficientes_lineares(expressao):
        ckp = float(expressao.coeff(Kp))
        cki = float(expressao.coeff(Ki))
        ckd = float(expressao.coeff(Kd))
        livre = float(expressao.subs({Kp: 0, Ki: 0, Kd: 0}))
        return [ckp, cki, ckd], livre

    # Linhas da Matriz
    row1, livre1 = extrair_coeficientes_lineares(coeffs_poly[1])  # z^3
    row2, livre2 = extrair_coeficientes_lineares(coeffs_poly[2])  # z^2
    row3, livre3 = extrair_coeficientes_lineares(coeffs_poly[3])  # z^1

    A = np.array([row1, row2, row3])

    # O vetor b é o coeficiente desejado MENOS o termo livre que passa pro outro lado
    b_vec = np.array([
        r_des[1] - livre1,
        r_des[2] - livre2,
        r_des[3] - livre3
    ])

    if debug:
        print("\nSistema Linear Montado [A]{x} = {b}:")
        print("Matriz A:")
        print(A)
        print("Vetor b:")
        print(b_vec)

    # Resolver
    x_gains = np.linalg.solve(A, b_vec)
    Kp_val, Ki_val, Kd_val = x_gains

    if debug:
        print("\n--- RESULTADO FINAL DOS GANHOS ---")
        print(f"Kp = {Kp_val:.5f}")
        print(f"Ki = {Ki_val:.5f}")
        print(f"Kd = {Kd_val:.5f}")

    # -----------------------------------------------------
    # 4. VERIFICAÇÃO DE ESTABILIDADE
    # -----------------------------------------------------
    if debug:
        print("\n[4] VERIFICAÇÃO (Critério de Jury / Raízes)")
        print("-" * 30)

    # Substituir ganhos na equação simbólica para obter polinômio final
    coeffs_numericos = []
    for c in coeffs_poly:
        val = c.subs({Kp: Kp_val, Ki: Ki_val, Kd: Kd_val})
        coeffs_numericos.append(float(val))

    raizes = np.roots(coeffs_numericos)
    estavel = all(np.abs(r) < 1 for r in raizes)

    if debug:
        print("Polinômio Final de Malha Fechada:")
        print(np.around(coeffs_numericos, 5))
        print(f"\nRaízes (Polos) obtidos:\n{raizes}")
        print(f"Módulos das raízes: {np.abs(raizes)}")
        print(f"\nO sistema é ESTÁVEL? {'>>> SIM <<<' if estavel else '>>> NÃO <<<'}")

        # -----------------------------------------------------
        # 5. PREPARAÇÃO PARA PARTE C
        # -----------------------------------------------------
        print("\n[5] PREPARAÇÃO PARA PARTE C")
        print("-" * 30)
        print(f"Guarde estes ganhos para o próximo script:")
        print(f"params = {{'Kp': {Kp_val:.4f}, 'Ki': {Ki_val:.4f}, 'Kd': {Kd_val:.4f}}}")

    else:
        # Modo silencioso: mostra apenas o essencial
        status = "ESTÁVEL" if estavel else "INSTÁVEL"
        print(f"Status: {status} | Polos Max: {np.max(np.abs(raizes)):.4f} | Kp={Kp_val:.4f}, Ki={Ki_val:.4f}, Kd={Kd_val:.4f}")

    return Kp_val, Ki_val, Kd_val


# Tentei de todas as formas utilizar um multiplo z3 entre 5 e 10, mas isso iria tornar o sistema LENTO com tempo de acomodação de quase 8 seg.
# Com ele em 37 fica 2.5seg de acomodação.
# if __name__ == "__main__":
#     ganhos_k(zeta_proj=0.5, wn_proj=3.0, multiplo_z3=37, z4=-0.65, debug=False)




# Jury
# \section*{Verificação Analítica de Estabilidade (Critério de Jury)}
#
# \subsection*{1. Equação Característica}
# Com base nos ganhos calculados ($K_p, K_i, K_d$), o polinômio de malha fechada obtido foi:
# \begin{equation}
#     D(z) = 1.0 z^4 - 2.8565 z^3 + 2.7173 z^2 - 0.8607 z - 0.0201 = 0
# \end{equation}
# Identificando os coeficientes ($a_n$ corresponde a $z^n$):
# \[
# a_4 = 1.0000, \quad a_3 = -2.8565, \quad a_2 = 2.7173, \quad a_1 = -0.8607, \quad a_0 = -0.0201
# \]
#
# \subsection*{2. Verificação das Condições Necessárias}
# Para que o sistema seja estável, todas as condições a seguir devem ser satisfeitas:
#
# \begin{itemize}
#     \item \textbf{Condição I:} $P(1) > 0$
#     Substituindo $z=1$ na equação característica:
#     \[
#     P(1) = 1.0 - 2.8565 + 2.7173 - 0.8607 - 0.0201
#     \]
#     \[
#     P(1) = -0.0200
#     \]
#     \[
#     \boxed{-0.0200 \ngtr 0} \implies \textbf{FALHOU}
#     \]
#
#     \item \textbf{Condição II:} $(-1)^4 P(-1) > 0$
#     \[
#     P(-1) = 1.0(1) - 2.8565(-1) + ... = 7.4144 > 0 \quad (\text{OK})
#     \]
#
#     \item \textbf{Condição III:} $|a_0| < a_4$
#     \[
#     |-0.0201| < 1.0000 \quad (\text{OK})
#     \]
# \end{itemize}
#
# \textit{Conclusão Parcial: O sistema já se mostra instável pela violação da Condição I.}
#
# \subsection*{3. Tabela de Jury (Demonstração Completa)}
# Apesar da falha na condição necessária, apresenta-se a tabela completa para fins de avaliação do método.
#
# \begin{table}[h!]
# \centering
# \renewcommand{\arraystretch}{1.3}
# \begin{tabular}{|c|c|c|c|c|c|}
# \hline
# \textbf{Linha} & \textbf{$z^0$} & \textbf{$z^1$} & \textbf{$z^2$} & \textbf{$z^3$} & \textbf{$z^4$} \\ \hline
# 1 & $a_0 = -0.0201$ & $a_1 = -0.8607$ & $a_2 = 2.7173$ & $a_3 = -2.8565$ & $a_4 = 1.0000$ \\ \hline
# 2 & $a_4 = 1.0000$ & $a_3 = -2.8565$ & $a_2 = 2.7173$ & $a_1 = -0.8607$ & $a_0 = -0.0201$ \\ \hline
# 3 & $b_0 = -0.9996$ & $b_1 = 2.8738$ & $b_2 = -2.7718$ & $b_3 = 0.9180$ & - \\ \hline
# 4 & $b_3 = 0.9180$ & $b_2 = -2.7718$ & $b_1 = 2.8738$ & $b_0 = -0.9996$ & - \\ \hline
# 5 & $c_0 = 0.1564$ & $c_1 = -0.3280$ & $c_2 = 0.1325$ & - & - \\ \hline
# \end{tabular}
# \caption{Tabela de Jury para $N=4$.}
# \end{table}
#
# \textbf{Cálculos Auxiliares (Determinantes):}
# \[ b_k = \begin{vmatrix} a_0 & a_{4-k} \\ a_4 & a_k \end{vmatrix}, \quad c_k = \begin{vmatrix} b_0 & b_{3-k} \\ b_3 & b_k \end{vmatrix} \]
# Exemplo: $b_0 = (-0.0201)^2 - (1.0)^2 = -0.9996$.
#
# \textbf{Análise das Condições Suficientes:}
# \begin{itemize}
#     \item $|b_0| > |b_3| \implies |-0.9996| > |0.9180|$ (Verdadeiro)
#     \item $|c_0| > |c_2| \implies |0.1564| > |0.1325|$ (Verdadeiro)
# \end{itemize}
#
# \subsection*{Conclusão Final}
# Apesar dos coeficientes internos da tabela satisfazerem as desigualdades de magnitude, o sistema é classificado como \textbf{INSTÁVEL} devido à violação da condição necessária $P(1) < 0$. Isso indica matematicamente a presença de pelo menos uma raiz real no semiplano direito (fora do círculo unitário), o que foi confirmado computacionalmente pela raiz em $z \approx 1.21$.