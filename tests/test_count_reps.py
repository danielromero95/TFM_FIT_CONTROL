# tests/test_count_reps.py

import pytest

from src.D_modeling.count_reps import count_reps_from_angles


def test_count_reps_simple_two_reps():
    """
    Secuencia sintética de ángulos de rodilla que simula 2 repeticiones completas:
      - Parte alta (>=160)
      - Baja (< 90)
      - Vuelve a alta (>=160) → 1 rep
      - Baja de nuevo y vuelve a alta → 2 repeticiones
    """
    angle_sequence = [
        170, 165, 150, 130, 100,  60,  80, 120, 150, 170,  # 1ª repetición
        165, 150, 130, 100,  50,  70, 120, 150, 170         # 2ª repetición
    ]
    reps = count_reps_from_angles(angle_sequence, low_thresh=90.0, high_thresh=160.0)
    assert reps == 2


def test_count_reps_no_cycles():
    """
    Casos donde nunca se forma un ciclo completo:
      - Nunca baja por debajo de low_thresh
      - Baja pero nunca sube nuevamente a high_thresh
    """
    # Nunca baja lo suficiente (< 90)
    seq_never_down = [170, 165, 160, 155, 150, 145]
    assert count_reps_from_angles(seq_never_down, low_thresh=90.0, high_thresh=160.0) == 0

    # Baja (<90) pero no sube a ≥160
    seq_never_up = [170, 165, 150, 130, 100,  60,  80, 120, 150, 155, 158]
    # Tras bajar a 60, solo sube a 155–158 (< 160) → no contamos nada
    assert count_reps_from_angles(seq_never_up, low_thresh=90.0, high_thresh=160.0) == 0


def test_count_reps_partial_cycles():
    """
    Caso mixto:
      1) Primera parte: baja (<90) pero no vuelve a >=160 antes de que la secuencia intermedia
         baje (< low_thresh) nuevamente → NO debe contar
      2) Después hay un ciclo real (baja <90 y sube >=160) → cuenta 1
    """
    seq = [
        170, 165, 130,  80,  85,   # Baja (<90); se encontraba en “up” (>=160) al principio
        100, 120, 130, 150,        # Zona intermedia (<160)
        155,  # Sube a 155 (<160) → no cuenta (sigue en “down”)
        158, 150, 130,  80,  50,   # Vuelve a bajar (< 90) sin haber llegado nunca a ≥160
        90, 150, 170               # Sube a 170 (>=160) → aquí sí contamos 1 rep completa
    ]
    reps = count_reps_from_angles(seq, low_thresh=90.0, high_thresh=160.0)
    assert reps == 1


def test_count_reps_custom_thresholds():
    """
    Probamos umbrales distintos (low_thresh=100, high_thresh=140):
      - Baja por debajo de 100, luego sube por encima de 140 → 1 rep
      - Baja nuevamente y sube → 2 repeticiones
    """
    seq = [
        150, 145, 140, 120,  95, 105, 130, 145, 150,  # 1ª repetición
        145, 140, 120,  90, 110, 130, 145            # 2ª repetición
    ]
    reps = count_reps_from_angles(seq, low_thresh=100.0, high_thresh=140.0)
    assert reps == 2
