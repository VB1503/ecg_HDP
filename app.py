import streamlit as st
from Ecg import  ECG
ecg = ECG()

st.set_page_config(
    page_title="Heart Disease Prediction ECG",
    page_icon="https://i.pngimg.me/thumb/f/720/comhiclipartcvbqn.jpg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
  st.title("Welcome to My Cardiography Prediction App")
  st.write("Upload your EGC here check your Herart disease risk")
  uploaded_file = st.file_uploader("Choose a file")

  if uploaded_file is not None:
    ecg_user_image_read = ecg.getImage(uploaded_file)
    st.image(ecg_user_image_read)
    ecg_user_gray_image_read = ecg.GrayImgae(ecg_user_image_read)
    my_expander = st.expander(label='Gray SCALE IMAGE')
    with my_expander: 
      st.image(ecg_user_gray_image_read)
    dividing_leads=ecg.DividingLeads(ecg_user_image_read)
    my_expander1 = st.expander(label='DIVIDING LEAD')
    with my_expander1:
      st.image('Leads_1-12_figure.png')
      st.image('Long_Lead_13_figure.png')
    ecg_preprocessed_leads = ecg.PreprocessingLeads(dividing_leads)
    my_expander2 = st.expander(label='PREPROCESSED LEAD')
    with my_expander2:
      st.image('Preprossed_Leads_1-12_figure.png')
      st.image('Preprossed_Leads_13_figure.png')
    ec_signal_extraction = ecg.SignalExtraction_Scaling(dividing_leads)
    my_expander3 = st.expander(label='CONOTUR LEADS')
    with my_expander3:
      st.image('Contour_Leads_1-12_figure.png')
    ecg_1dsignal = ecg.CombineConvert1Dsignal()
    my_expander4 = st.expander(label='1D Signals')
    with my_expander4:
      st.write(ecg_1dsignal)
    ecg_final = ecg.DimensionalReduciton(ecg_1dsignal)
    my_expander4 = st.expander(label='Dimensional Reduction')
    with my_expander4:
      st.write(ecg_final)
    ecg_model=ecg.ModelLoad_predict(ecg_final)
    my_expander5 = st.expander(label='PREDICTION')
    with my_expander5:
      st.write(ecg_model)

def about():
    st.title("About Our Project")

    # Project Information
    st.header(f"PROJECT TITLE: DETECTION OF CARDIOVASCULAR DISEASES IN ECG IMAGES USING MACHINE LEARNING AND DEEP LEARNING METHODS")


    # College Information
    st.header("College Name:")
    st.write("University College Of Engineering Kanchipuram")

    # Batch Members Information
    st.header("Batch Members:")
    batch_members = [
        ("GOKULA KRISHNAN B ", "513420104008"),
        ("SARAN S", "513420103009"),
        ("JAYANTH G", "513420104703"),
        ("BARANIDHARAN T", "513420104704")
    ]
    
    for member_name, roll_number in batch_members:
        st.write(f"{member_name} ({roll_number})")

    # Guided By Information
    st.header("Guided By:")
    st.write("Mr. J. DEVANATHAN, Teaching Fellow")

    # Project Proposed Work
    st.header("Project Proposed Work:")
    st.markdown("The comprehensive nature of the proposed system underscores its significance in revolutionizing the diagnostic process for cardiovascular problems. In the conventional healthcare setting, the interpretation of electrocardiogram (ECG) charts demands extensive training, while the manual examination of paper records proves to be both time-consuming and laborious. The primary objective of this project is to harness the power of machine learning methodologies to streamline and automate the conversion of traditional paper-based ECG records into efficient digital 1-D signals. This transformative approach is geared towards enhancing the accuracy and efficiency of cardiovascular diagnosis.")
    st.markdown("The intricate process begins with the extraction of pivotal components such as the P, QRS, and T waves from the ECG signals. The original ECG report is systematically divided into 13 leads, each subsequently transformed into digital signals. To ensure signal quality, smoothing techniques are meticulously applied. The signals then undergo a transformation into binary images using advanced thresholding methods. Feature extraction assumes a crucial role in this process, with dimension reduction techniques like Principal Component Analysis (PCA) employed to gain a nuanced understanding of the data and capture pertinent information.")
    st.markdown("At the heart of the proposed system lies the implementation of multiple machine learning classifiers, namely k-nearest neighbors (KNN), Logistic Regression, Support Vector Machine (SVM), and a Voting-Based Ensemble Classifier. These classifiers are diligently trained to predict cardiac conditions based on the extracted features from the digitized ECG signals. Rigorous evaluation is conducted using key metrics such as accuracy, precision, recall, F1-score, and support. The final selection of the model hinges on meeting predefined criteria for these metrics, thereby ensuring the reliability and efficacy of the diagnostic tool.")
    st.markdown("In practical terms, the system's objective is to diagnose patients for specific cardiac conditions, with a focus on the detection of Myocardial Infarction, identification of Abnormal Heartbeat patterns, and affirmation of a healthy cardiac state. The automation of the ECG analysis process brings forth numerous advantages, including heightened efficiency, a reduction in manual effort, and early detection of potential cardiac issues. The anticipated outcomes of the proposed system encompass accurate and consistent diagnoses, improved accessibility to ECG analysis, and the establishment of a streamlined approach to cardiovascular health assessment.")
nav_bar = st.container()

# Add elements to the navigation bar
with nav_bar:
    menu = ["Home", "About"]
    selected_page = st.radio("Go to", menu)

# Display the selected page
if selected_page == "Home":
    main()
elif selected_page == "About":
    about()
