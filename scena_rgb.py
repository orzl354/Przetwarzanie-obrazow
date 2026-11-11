"""
Moduł do generowania przykładowej sceny RGB na potrzeby laboratorium WPOB.

Kolory należy podawać w kolejności RGB, jako krotki (R, G, B)
z wartościami całkowitymi z zakresu 0–255.
"""

import numpy as np
import cv2


def _dodaj_obiekt(obraz, x, y, srodek, promien, kolor,
                  wsp_gradientu=0.0035, min_jasnosc=0.62):
    """Rysuje kołowy obiekt z łagodnym gradientem jasności."""
    cx, cy = srodek

    odleglosc = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    maska = odleglosc < promien

    jasnosc = np.clip(
        1.0 - wsp_gradientu * odleglosc,
        min_jasnosc,
        1.0
    )

    for k in range(3):
        kanal = obraz[:, :, k].astype(np.float32)
        kanal[maska] = kolor[k] * jasnosc[maska]
        obraz[:, :, k] = np.clip(kanal, 0, 255).astype(np.uint8)


def generuj_scene(kolory, wysokosc=360, szerokosc=520,
                  szum=5, ziarno=7):
    """
    Generuje scenę RGB z zielonym tłem, dwoma liśćmi i dwoma obiektami.

    Parametry
    ---------
    kolory : dict
        Słownik zawierający klucze:
        'tlo', 'lisc1', 'lisc2', 'obiekt1', 'obiekt2',
        'refleks1', 'refleks2'.
        Każdy kolor jest krotką (R, G, B), wartości 0–255.
    wysokosc, szerokosc : int
        Wymiary generowanego obrazu.
    szum : float
        Odchylenie standardowe szumu gaussowskiego. Ustaw 0, aby wyłączyć szum.
    ziarno : int lub None
        Ziarno generatora liczb losowych; zapewnia powtarzalność obrazu.

    Zwraca
    -------
    numpy.ndarray
        Obraz RGB typu uint8 o wymiarach (wysokosc, szerokosc, 3).
    """

    wymagane_kolory = (
        "tlo", "lisc1", "lisc2", "obiekt1", "obiekt2",
        "refleks1", "refleks2"
    )

    brakujace = [nazwa for nazwa in wymagane_kolory if nazwa not in kolory]
    if brakujace:
        raise ValueError(
            "Brakuje kolorów w słowniku: " + ", ".join(brakujace)
        )

    for nazwa in wymagane_kolory:
        kolor = kolory[nazwa]
        if len(kolor) != 3 or any(not 0 <= wartosc <= 255 for wartosc in kolor):
            raise ValueError(
                f"Kolor '{nazwa}' musi zawierać trzy wartości RGB z zakresu 0–255."
            )

    y, x = np.mgrid[0:wysokosc, 0:szerokosc]
    obraz = np.zeros((wysokosc, szerokosc, 3), dtype=np.uint8)

    # Tło z łagodnym gradientem zachowującym charakter sceny.
    obraz[:, :, 0] = np.clip(
        kolory["tlo"][0] + y * 15 / wysokosc, 0, 255
    ).astype(np.uint8)
    obraz[:, :, 1] = np.clip(
        kolory["tlo"][1] + x * 35 / szerokosc, 0, 255
    ).astype(np.uint8)
    obraz[:, :, 2] = np.clip(
        kolory["tlo"][2] + x * 12 / szerokosc, 0, 255
    ).astype(np.uint8)

    # Liście.
    cv2.ellipse(
        obraz, (125, 125), (90, 35), -25, 0, 360,
        tuple(map(int, kolory["lisc1"])), -1
    )
    cv2.ellipse(
        obraz, (390, 245), (100, 32), 25, 0, 360,
        tuple(map(int, kolory["lisc2"])), -1
    )

    # Niebieski / drugi obiekt z gradientem jasności.
    _dodaj_obiekt(
        obraz, x, y,
        srodek=(105, 270),
        promien=48,
        kolor=kolory["obiekt2"],
        wsp_gradientu=0.004,
        min_jasnosc=0.65
    )
    cv2.ellipse(
        obraz, (90, 255), (10, 20), 25, 0, 360,
        tuple(map(int, kolory["refleks2"])), -1
    )

    # Czerwony / pierwszy obiekt z gradientem jasności.
    _dodaj_obiekt(
        obraz, x, y,
        srodek=(270, 180),
        promien=92,
        kolor=kolory["obiekt1"],
        wsp_gradientu=0.0035,
        min_jasnosc=0.62
    )
    cv2.ellipse(
        obraz, (238, 145), (18, 35), 25, 0, 360,
        tuple(map(int, kolory["refleks1"])), -1
    )

    # Szum dodawany na końcu do całego obrazu.
    if szum > 0:
        rng = np.random.default_rng(ziarno)
        obraz = np.clip(
            obraz.astype(np.float32)
            + rng.normal(0, szum, obraz.shape),
            0, 255
        ).astype(np.uint8)

    return obraz
