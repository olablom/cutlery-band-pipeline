# 📊 Statusrapport – 2025-11-06

## Projekt: Cutlery Band Pipeline (Pi + Hailo)

**Sammanfattning:**

Typklassificeringen är nu fullständigt verifierad end-to-end på Raspberry Pi 5 med Hailo-8-accelerator.

Inferens fungerar stabilt i realtid (≈ 1.5–1.9 ms per bild), loggning och rapportstruktur är etablerade, och variant-delen är tekniskt färdig men väntar på nya bilddata från riktiga tillverkare.

---

## ✅ Genomfört

### Systemnivå

* Full repo-struktur etablerad (`acs-runtime`, `deployment`, `dataset`, `scripts`, `reports`).

* Samma kodbas körs oförändrad på PC och Pi.

* Stabil HEF-inferenz via `run_hailo_variant_loop.py` (PCIe, HailoRT).

* CSV- och textloggning aktiv i `reports/` för varje körning.

### Prestanda

| Miljö          | Backend                      | Mean (ms)  | P95 (ms) | Kommentar          |
| -------------- | ---------------------------- | ---------- | -------- | ------------------ |
| Pi 5 CPU       | ONNX Runtime                 | 23 ms      | 33 ms    | Stabil baseline    |
| Pi 5 + Hailo-8 | HailoRT (HEF)                | 1.5–1.9 ms | < 2 ms   | Produktionsklar    |
| PC GPU         | CUDA (CUDAExecutionProvider) | 0.7 ms     | 1.0 ms   | Valideringsträning |

### Modell

* Aktiv modell: `type_classifier_480x170_single_fixed.hef`

* Arkitektur: SqueezeNet 1.1

* Dataset: 1 500+ bilder, 80/20 split, full loggning.

* Accuracy på typnivå: 100 % på golden set.

---

## ⚙️ Pågående arbete (6–7 nov)

### Torsdag

* Rensning av Pi-miljö, borttagning av dubbletter.

* Kall-boot-test av pipeline.

* Tidsloggning till CSV aktiverad.

* Förberedelse av variant-datasetstruktur:

  ```
  dataset/variants/{fork,knife,spoon}/{ikea_365,wmf_basic,stelton_classic}
  ```

* Kontroll av CUDA och checkpoints i `variant_train_extract.py`.

### Fredag

* Fotografering av riktiga tillverkare.

* Import och strukturering av variantbilder.

* Första träning av variant-modell och prototyp-export.

* Test av end-to-end variant-inferens på Pi.

---

## 🚀 Nästa steg

1. **Variant-dataset:** fotografera och lägga till IKEA 365, WMF Basic och Stelton Classic.

2. **Träning:** köra `variant_train_extract.py` med riktiga data, generera nya prototyper.

3. **Inferens:** testa variant-pipeline på Pi, mäta accuracy och score-separation.

4. **Integration:** koppla AI-modulen till PLC-styrning när maskinen anländer (v. 4).

5. **Övervakning:** logga långtidsprestanda (CPU-last, temperatur, latensdrift).

---

*Här är vi nu*

