**Κεφάλαιο 1: Εισαγωγή**
1.1 Τεχνητή νοημοσύνη στην υγεία
1.2 Το πρόβλημα του "μαύρου κουτιού"
1.3 Επεξηγήσιμη τεχνητή νοημοσύνη (XAI)
1.4 Στόχοι και δομή της εργασίας ← this is where you motivate the two-dataset, two-track approach in one paragraph

**Κεφάλαιο 2: Θεωρητικό υπόβαθρο**
2.1 Random Forests
2.2 Deep Neural Networks (DNNs)
2.3 Convolutional Neural Networks (CNNs)
2.4 Vision Transformers (ViT)
2.5 Τεχνικές ερμηνευσιμότητας
— 2.5.1 SHAP
— 2.5.2 LIME
— 2.5.3 Grad-CAM

Note: open 2.5 with a sentence distinguishing post-hoc attribution methods (SHAP, LIME) from gradient-based visualisation (Grad-CAM). One sentence is enough; it pre-empts the examiner doing it for you.

**Κεφάλαιο 3: Δεδομένα και προεπεξεργασία**
3.1 Cleveland Heart Disease Dataset — ανάλυση και χαρακτηριστικά
3.2 Προεπεξεργασία αριθμητικών δεδομένων
3.3 HAM10000 Dataset — ανάλυση και κατανομή κλάσεων
3.4 Προεπεξεργασία εικόνων και διαχείριση ανισορροπίας κλάσεων ← imbalance handling belongs here, not buried in Chapter 4

**Κεφάλαιο 4: Εκπαίδευση μοντέλων και εφαρμογή XAI**
4.1 Πρόβλεψη καρδιοπαθειών
— 4.1.1 Αρχιτεκτονική και εκπαίδευση RF & DNN
— 4.1.2 Αποτελέσματα ταξινόμησης
— 4.1.3 Ερμηνεία με SHAP & LIME

4.2 Διάγνωση δερματικών αλλοιώσεων
— 4.2.1 Αρχιτεκτονική και εκπαίδευση CNN & ViT
— 4.2.2 Αποτελέσματα ταξινόμησης
— 4.2.3 Οπτικοποίηση με Grad-CAM

**Κεφάλαιο 5: Συζήτηση**
5.1 Σύγκριση μοντέλων και μεθόδων ερμηνείας ← cross-track comparison lives here, not scattered across Ch4
5.2 Κλινική σημασία των αποτελεσμάτων

**Κεφάλαιο 6: Συμπεράσματα**
6.1 Συμπεράσματα
6.2 Περιορισμοί
6.3 Μελλοντική εργασία

One unresolved question: the clinical evaluation by health professionals was in your original methodology and has been dropped without explanation. If you're not doing it, 6.2 needs a sentence acknowledging its absence and why (sample size, access, time). If you are doing it, it needs a section in Chapter 4 or 5.
