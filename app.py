import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import zscore
import joblib

# Set halaman Streamlit
st.set_page_config(page_title="Klasifikasi Penyakit Serangan Jantung Menggunakan Metode Decision Tree C45", layout="wide")

# Fungsi untuk memuat dataset
@st.cache_data
def load_data():
    return pd.read_csv('heart.csv')

# Memuat dataset
df = load_data()

# Sidebar navigasi
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Analisis Data", "Pre Processing", "Modelling", "Klasifikasi"])

# Analisis Data
if menu == "Analisis Data":
    st.title("Analisis Data - Klasifikasi Penyakit Serangan Jantung Menggunakan Metode Decision Tree C45")
    
    st.markdown("""
    **Nama**: Mochammad Mahreza Rizky Fahrozi  
    **NIM**: 220411100069  
    **Kelas**: Proyek Sain Data IF5B
    """)

    # Penjelasan fitur dan target
    st.markdown("Penjelasan Fitur dan Target dalam Dataset Penyakit Jantung")

    st.markdown("""
    Berikut penjelasan lebih rinci untuk setiap atribut dalam dataset ini:

    1. **age (umur)**: Menggambarkan usia individu dalam tahun. Umur merupakan faktor risiko yang signifikan untuk penyakit jantung, di mana risiko biasanya meningkat seiring bertambahnya usia.

    2. **sex (jenis kelamin)**: Jenis kelamin individu, dinyatakan sebagai angka biner (1 = laki-laki, 0 = perempuan). Jenis kelamin memengaruhi risiko penyakit jantung, di mana pria cenderung memiliki risiko yang lebih tinggi dibandingkan wanita.

    3. **cp (jenis nyeri dada)**: Menunjukkan tipe nyeri dada yang dialami individu, yang bisa memberi indikasi tentang risiko penyakit jantung:
    - 0: **Angina tipikal** – Nyeri dada yang sering terjadi akibat penyempitan arteri.
    - 1: **Angina atipikal** – Nyeri dada yang mungkin tidak terkait langsung dengan penyempitan arteri.
    - 2: **Nyeri non-angina** – Nyeri dada yang tidak berhubungan dengan penyakit jantung.
    - 3: **Tanpa gejala** – Individu tidak mengalami nyeri dada tetapi mungkin memiliki faktor risiko lain.

    4. **trtbps (tekanan darah saat istirahat)**: Tekanan darah saat istirahat (dalam mm Hg) selama pemeriksaan, yang bisa menunjukkan risiko hipertensi, yang juga faktor risiko utama penyakit jantung.

    5. **chol (kolesterol)**: Tingkat kolesterol total dalam serum darah (mg/dl). Kolesterol tinggi dapat meningkatkan risiko penumpukan plak pada arteri, sehingga meningkatkan risiko penyakit jantung.

    6. **fbs (gula darah puasa)**: Tingkat gula darah setelah puasa (lebih dari 120 mg/dl dinyatakan sebagai "benar" atau abnormal; sebaliknya, "salah" atau normal) yang ditandai dengan nilai biner (1 = benar, 0 = salah). Gula darah tinggi bisa menjadi indikasi diabetes, yang merupakan faktor risiko tambahan.

    7. **restecg (hasil elektrokardiografi saat istirahat)**: Hasil EKG untuk menunjukkan apakah ada aktivitas jantung abnormal:
    - 0: **Normal** – Tidak ada indikasi kelainan.
    - 1: **Abnormalitas gelombang ST-T** – Perubahan pada gelombang ST atau T, yang dapat menunjukkan adanya masalah pada otot jantung atau suplai darah.
    - 2: **Hipertrofi ventrikel kiri** – Pembesaran dinding jantung bagian kiri yang bisa terkait dengan tekanan darah tinggi.

    8. **thalachh (detak jantung maksimum yang dicapai)**: Detak jantung maksimum yang dicapai individu selama uji jantung. Detak jantung yang tinggi saat latihan fisik bisa menandakan bahwa jantung bekerja keras untuk memompa darah, yang dapat digunakan untuk mengevaluasi risiko kardiovaskular.

    9. **exng (angina yang diinduksi olahraga)**: Menunjukkan apakah angina (nyeri dada) muncul saat melakukan olahraga atau aktivitas fisik (1 = ya, 0 = tidak). Angina yang terinduksi oleh olahraga bisa menunjukkan penyempitan arteri koroner.

    10. **oldpeak**: Merupakan depresi segmen ST yang diukur saat olahraga relatif terhadap keadaan istirahat. Depresi ST menunjukkan kurangnya suplai oksigen ke otot jantung.

    11. **slp (kemiringan segmen ST puncak saat olahraga)**: Bentuk atau arah segmen ST selama aktivitas fisik:
        - 0: **Menanjak** – Biasanya dianggap normal.
        - 1: **Datar** – Bisa menunjukkan iskemia atau kekurangan suplai darah.
        - 2: **Menurun** – Lebih sering dikaitkan dengan risiko tinggi terhadap penyakit jantung.

    12. **caa (jumlah pembuluh besar yang terlihat melalui fluoroskopi)**: Jumlah pembuluh darah besar (dari 0 hingga 3) yang terlihat saat pemeriksaan fluoroskopi, yang menunjukkan seberapa banyak pembuluh darah utama jantung yang terbuka atau tidak tersumbat.

    13. **thall (status thalasemia)**: Menunjukkan status talasemia atau jenis hemoglobin yang dibawa individu:
        - 1: **Normal**
        - 2: **Cacat tetap** – Talasemia tetap yang bisa membatasi kemampuan darah membawa oksigen.
        - 3: **Cacat reversibel** – Talasemia yang bersifat sementara atau tidak permanen.

    14. **output**: Variabel target yang menunjukkan adanya penyakit jantung (1) atau tidak (0).
    """)

    
    # Tampilan 5 Data Awal
    st.write("### Tampilan 5 Data Awal")
    st.dataframe(df.head())

    # Informasi Atribut Dataset
    st.write("### Informasi Atribut Dataset")
    buffer = df.info()
    st.text(buffer)

    # Analisis Statistik
    st.write("### Analisis Statistik")
    st.dataframe(df.describe())

    # Distribusi Target
    st.write("### Distribusi Target")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='output', data=df)
    plt.title("Distribution of Stroke Outcome")
    plt.xlabel("Stroke Outcome")
    plt.ylabel("Count")
    st.pyplot(plt)

    stroke_counts = df['output'].value_counts()
    stroke_distribution = df['output'].value_counts(normalize=True) * 100
    stroke_summary = pd.DataFrame({
        'Count': stroke_counts,
        'Percentage (%)': stroke_distribution
    })
    st.write("Distribusi Target:")
    st.table(stroke_summary)

    # Distribusi Fitur Numerik
    st.write("### Distribusi Fitur Numerik")
    numerical_features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
    df[numerical_features].hist(bins=20, figsize=(15, 6), layout=(1, 5))
    plt.suptitle("Distribusi Fitur Numerik")
    st.pyplot(plt)

    # Distribusi Fitur Kategorikal
    st.write("### Distribusi Fitur Kategorikal")
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']
    df[categorical_features].hist(bins=20, figsize=(15, 6), layout=(2, 4))
    plt.suptitle("Distribusi Fitur Kategorikal")
    st.pyplot(plt)
    





# Pre Processing
elif menu == "Pre Processing":
    st.title("Pre Processing - Klasifikasi Penyakit Serangan Jantung Menggunakan Metode Decision Tree C45")
    
    st.markdown("""
    **Nama**: Mochammad Mahreza Rizky Fahrozi  
    **NIM**: 220411100069  
    **Kelas**: Proyek Sain Data IF5B
    """)
    # Mengecek Missing Value
    st.write("### Mengecek Missing Values")
    missing_values = df.isnull().sum()
    st.table(missing_values[missing_values > 0])

    # Mengecek Outliers Dengan Z-Score
    st.write("### Mengecek Outliers Dengan Z-Score")
    z_scores = np.abs(zscore(df.select_dtypes(include=['float64', 'int64'])))
    outliers = (z_scores > 3)
    outliers_summary = pd.DataFrame(outliers, columns=df.select_dtypes(include=['float64', 'int64']).columns)
    st.table(outliers_summary.any())

    # Visualisasi Outliers
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    st.title("Visualisasi Outliers Dengan Z-Score")
    for col in numeric_columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(range(len(df)), df[col], label='Data', alpha=0.7)
        ax.scatter(df.index[outliers[col]], df[col][outliers[col]], color='red', label='Outliers', alpha=0.7)
        ax.set_title(f"Outliers in {col} (Z-score > 3)")
        ax.set_xlabel("Index")
        ax.set_ylabel(col)
        ax.legend()
        st.pyplot(fig)

    # Mengganti Outliers Dengan Batas Persentil
    st.write("### Mengganti Outliers Dengan Batas Persentil")
    for col in ['trtbps', 'chol', 'thalachh', 'oldpeak', 'caa', 'thall']:
        lower_limit = df[col].quantile(0.05)
        upper_limit = df[col].quantile(0.95)
        df[col] = np.where(df[col] < lower_limit, lower_limit, df[col])
        df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])

    st.write("Outliers berhasil diganti dengan batas persentil 5% dan 95%.")

    # Mengecek Outliers Dengan Z-Score Setelah Diatasi
    st.write("### Mengecek Outliers Dengan Z-Score Setelah Diatasi")
    z_scores = np.abs(zscore(df.select_dtypes(include=['float64', 'int64'])))
    outliers = (z_scores > 3)
    outliers_summary = pd.DataFrame(outliers, columns=df.select_dtypes(include=['float64', 'int64']).columns)
    st.table(outliers_summary.any())

    # Visualisasi Outliers Setelah Diatasi
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    st.title("Visualisasi Outliers Setelah Diatasi")
    for col in numeric_columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(range(len(df)), df[col], label='Data', alpha=0.7)
        ax.scatter(df.index[outliers[col]], df[col][outliers[col]], color='red', label='Outliers', alpha=0.7)
        ax.set_title(f"Outliers in {col} (Z-score > 3)")
        ax.set_xlabel("Index")
        ax.set_ylabel(col)
        ax.legend()
        st.pyplot(fig)


    # Memisahkan Fitur Dan Target
    st.write("### Memisahkan Fitur dan Target")
    X = df.drop('output', axis=1)  # Fitur
    y = df['output']  # Label (target)
    st.write(f"Jumlah Fitur (X): {X.shape[1]}")
    st.write(f"Jumlah Target (y): {y.shape[0]}")



    # Visualisasi Pembagian Fitur Dan Target
    st.title("Visualisasi Pembagian Fitur Dan Target")
    features = X.columns
    target = ['output']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(features, [1] * len(features), color='skyblue', label='Features')
    ax.barh(target, [1], color='orange', label='Target')
    ax.set_title("Feature and Target Attributes")
    ax.set_xlabel("Presence (1 indicates attribute type)")
    ax.set_yticks(range(len(features) + 1))
    ax.set_yticklabels(list(features) + target)
    ax.legend()
    st.pyplot(fig)


    # Data Splitting
    st.write("### Membagi Data (Train-Test Split)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write(f"Jumlah Data Training: {X_train.shape[0]} rows, Jumlah Fitur: {X_train.shape[1]}")
    st.write(f"Jumlah Data Testing: {X_test.shape[0]} rows")


    # Visualisasi Data Splitting
    st.title("Visualisasi Data Splitting")
    train_size = len(X_train)
    test_size = len(X_test)

    categories = ['Training Set', 'Test Set']
    counts = [train_size, test_size]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(categories, counts, color=['skyblue', 'orange'])
    ax.set_title("Dataset Sizes: Training vs Testing")
    ax.set_ylabel("Number of Samples")
    ax.text(0, train_size + 5, f'{train_size} samples', ha='center', va='bottom', fontsize=10)
    ax.text(1, test_size + 5, f'{test_size} samples', ha='center', va='bottom', fontsize=10)
    st.pyplot(fig)

    # Keterangan dalam teks
    st.write(f"Jumlah data untuk Training Set: {train_size} sampel")
    st.write(f"Jumlah data untuk Test Set: {test_size} sampel")

    # Distribusi Target di Data Training dan Testing
    st.write("### Distribusi Target di Data Training dan Testing")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Distribusi Training
    sns.countplot(x=y_train, data=df, ax=ax[0])
    ax[0].set_title("Distribusi Target Training Data")
    ax[0].set_xlabel("Stroke Outcome")
    ax[0].set_ylabel("Count")

    # Distribusi Testing
    sns.countplot(x=y_test, data=df, ax=ax[1])
    ax[1].set_title("Distribusi Target Testing Data")
    ax[1].set_xlabel("Stroke Outcome")
    ax[1].set_ylabel("Count")
    st.pyplot(fig)


    # Menghitung distribusi target di training set dan test set
    train_dist = y_train.value_counts().reset_index()
    train_dist.columns = ['output', 'count']

    test_dist = y_test.value_counts().reset_index()
    test_dist.columns = ['output', 'count']

    # Menampilkan distribusi target di training set
    st.write("Distribusi target di training set:")
    st.write(train_dist)

    # Menampilkan distribusi target di test set
    st.write("Distribusi target di test set:")
    st.write(test_dist)


# Modelling
elif menu == "Modelling":
    st.title("Modelling - Klasifikasi Penyakit Serangan Jantung Menggunakan Metode Decision Tree C45")
    
    st.markdown("""
    **Nama**: Mochammad Mahreza Rizky Fahrozi  
    **NIM**: 220411100069  
    **Kelas**: Proyek Sain Data IF5B
    """)

    # Memisahkan Fitur Dan Target
    X = df.drop('output', axis=1)  # Fitur
    y = df['output']  # Label (target)
    
    # Membagi Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.write(f"Training Data: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    st.write(f"Testing Data: {X_test.shape[0]} rows")

    # Membangun Model Decision Tree
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=10)
    clf.fit(X_train, y_train)

    # Menyimpan model setelah pelatihan
    joblib.dump(clf, 'heart_attack_model.pkl')  # Simpan model

    # Evaluasi Model
    y_pred = clf.predict(X_test)
    
    st.write("### Evaluasi Model")
    st.write(f"**Accuracy**: {accuracy_score(y_test, y_pred):.2f}")
    
    # Menampilkan confusion matrix dalam bentuk heatmap
    st.write("**Confusion Matrix**:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write(conf_matrix)
    
    # Menampilkan classification report
    st.write("**Classification Report**:")
    report = classification_report(y_test, y_pred)
    st.text(report)
    
    # Visualisasi Pohon Keputusan
    st.write("### Visualisasi Pohon Keputusan")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No Disease', 'Disease'], rounded=True, ax=ax)
    st.pyplot(fig)

# Klasifikasi
elif menu == "Klasifikasi":
    st.title("Klasifikasi - Input")
    
    st.markdown("""
    **Nama**: Mochammad Mahreza Rizky Fahrozi  
    **NIM**: 220411100069  
    **Kelas**: Proyek Sain Data IF5B
    """)

    # Muat model yang sudah dilatih
    clf = joblib.load('heart_attack_model.pkl')  # Memuat model yang disimpan sebelumnya
    
    # Masukkan data untuk prediksi
    st.write("Masukkan data untuk prediksi:")
    input_data = {}
    categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']  # Adjust categorical columns based on your dataset

    # Handle categorical and numerical features separately
    for col in df.drop('output', axis=1).columns:
        if col in categorical_columns:
            # Categorical input with selectbox
            categories = df[col].unique().tolist()
            input_data[col] = st.selectbox(f"{col}:", categories, index=categories.index(df[col].mode()[0]))
        else:
            # Numerical input with number_input and initial value set to None (0 or empty)
            input_data[col] = st.number_input(f"{col}:", value=0.0)  # Initial value 0.0 as a placeholder

    if st.button("Prediksi"):
        # Membuat DataFrame dari input pengguna
        input_df = pd.DataFrame([input_data])
        
        # Prediksi menggunakan model yang dimuat
        prediction = clf.predict(input_df)[0]
        
        # Menampilkan hasil prediksi
        result = "Disease" if prediction == 1 else "No Disease"
        st.write(f"Hasil Prediksi: {result}")
