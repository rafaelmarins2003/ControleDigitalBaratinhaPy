#include <Arduino.h>
#include <Baratinha.h>
#include <math.h>

Baratinha bra; // Instancia unica do robo Baratinha

namespace { // Escopo anonimo para constantes e estados locais
  // ======================================
  // PARAMETROS DO CONTROLE
  // ======================================
  // ATENCAO: mesmo Ts usado na parte teórica/simulação: 0,01 s
  const float Ts = 0.01f; // Periodo de controle em segundos (10 ms)

  // Ganhos finais da Parte C
  const float Kp = 0.004374066092882029f;
  const float Ki = 0.00844489305954211f;
  const float Kd = 2.2236406200986116f;

  // Filtro do termo derivativo (0 < alpha < 1)
  const float derivAlpha = 0.9999999999999f;

  // Saturacao em termos de PWM (Baratinha usa 8 bits: -255 a 255)
  const float Umax = 255.0f;
  const float Umin = -255.0f;

  // Segurança e referência
  const float distSeguranca_cm = 30.0f;   // desliga motores se < 3 cm
  const float setpoint_cm      = 100.0f;  // ex: manter 10 cm do obstáculo

  // Estados do PID
  float erro       = 0.0f;
  float erro_ant   = 0.0f;
  float integral   = 0.0f;
  float deriv_filt = 0.0f;

  // --------------------------------------
  // Função para zerar estados do controlador
  // --------------------------------------
  void resetControl() {
    erro       = 0.0f;
    erro_ant   = 0.0f;
    integral   = 0.0f;
    deriv_filt = 0.0f;
  }

  // --------------------------------------
  // Cálculo do PID discreto (uma iteração)
  // Entrada: referência e medida em cm
  // Saída: sinal de controle já saturado em [-Umax, Umax]
  // --------------------------------------
  float calculaPID(float referencia_cm, float medida_cm) {
    // Erro
    erro = referencia_cm - medida_cm;

    // Derivada bruta do erro
    float d_raw = (erro - erro_ant) / Ts;

    // Filtro exponencial no termo derivativo
    deriv_filt = derivAlpha * deriv_filt
               + (1.0f - derivAlpha) * d_raw;

    // Integral candidata
    float integral_cand = integral + erro * Ts;

    // Controle sem saturacao
    float u_unsat = Kp * erro
                  + Ki * integral_cand
                  + Kd * deriv_filt;

    // Saturacao
    float u_sat = constrain(u_unsat,-1,1);

    // Anti-windup (integral condicionado)
    bool saturou_alto  = (u_unsat > 1);
    bool saturou_baixo = (u_unsat < -1);

    bool libera_integral =
        (!saturou_alto && !saturou_baixo) ||      // não saturou
        (saturou_alto  && erro < 0.0f)      ||    // saturado alto, erro ajuda a diminuir
        (saturou_baixo && erro > 0.0f);           // saturado baixo, erro ajuda a aumentar

    if (libera_integral) {
      integral = integral_cand;
    }

    // Atualiza histórico
    erro_ant = erro;

    return u_sat;
  }

} // fim do namespace anônimo

void setup() {
  bra.recoveryMode(); // Modo de recuperacao (verifica botao para setup WiFi)
  bra.setupAll();     // Configuracao completa do hardware padrao

  bra.setControlInterval(Ts); // Define o periodo de controle (10 ms)
  bra.awaitStart();           // Aguarda o toque no botao para iniciar

  resetControl();             // Garante estados zerados no início

  bra.println("Controle PID discreto iniciado (Parte D)");
}

void loop() {
  bra.updateStartStop();         // Atualiza estado de start/stop
  if (!bra.isRunning()) return;  // Sai se nao estiver executando
  if (!bra.controlTickDue()) return; // Sai se nao for hora do proximo ciclo

  // ======================================
  // CICLO DE CONTROLE
  // ======================================

  // 1) Leitura da distância (ToF retorna em mm)
  float dist_mm = bra.readDistance();
  float dist_cm = dist_mm; // converte mm -> cm

  // 2) Segurança: se muito perto, desliga motores e reseta controlador
  if (dist_cm < distSeguranca_cm) {
    bra.stop();
    resetControl();
    bra.printf("SEGURANCA: dist = %.2f cm, motores desligados.\n", dist_cm);
    return;
  }

  // 3) Cálculo do PID (referência e medida em cm)
  float u = calculaPID(setpoint_cm, dist_cm);

  // 4) Aplica controle nos motores (u já está saturado em [-255, 255])
  int pwm = u*255;   // converte float -> int

  // move1D: positivo anda para frente, negativo para trás
  bra.move1D(-pwm);

  // (Opcional) debug na serial
  // bra.printf("dist=%.2f cm, ref=%.2f cm, erro=%.2f, u=%.2f, pwm=%d\n",
  //            dist_cm, setpoint_cm, erro, u, pwm);
}

/*
Métodos úteis da classe Baratinha:
- bool setupAll() : Inicializa todos os componentes de hardware.
- void setControlInterval(float seconds) : Define o intervalo de controle em segundos.
- bool controlTickDue() : Verifica se é hora de executar o próximo ciclo de controle
- void println(const char* msg) : Imprime uma mensagem na porta serial.
- void printf(const char* format, ...) : Imprime uma mensagem formatada na porta serial
- bool isRunning() : Verifica se o robô está em modo de execução.
- void updateStartStop() : Atualiza o estado de start/stop com base no botão.
- void move1D(int pwm, bool light = false) : Move o robô em uma direção com velocidade especificada.
- void stop() : Para os motores imediatamente.
- float readDistance() : Lê a distância do sensor ToF em milímetros.
*/
