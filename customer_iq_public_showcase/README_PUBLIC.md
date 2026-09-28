# CustomerIQ — Public Hackathon Edition

**CustomerIQ turns raw customer data into an interactive customer galaxy.**

Upload CSV, Excel or SQLite data, train a segmentation model, explore customers in 3D, simulate a new customer and translate each segment into a business playbook.

## Highlights
- Dynamic visual themes: Aurora, Cyber, Sunset
- Animated 3D customer-galaxy hero using Three.js
- RFM + behavioral segmentation
- K-Means with Elbow, Silhouette, Davies-Bouldin and Calinski-Harabasz diagnostics
- Interactive 2D/3D PCA cluster exploration
- Real-time new-customer classifier
- Business personas and action playbooks
- No Snowflake dependency
- Streamlit Community Cloud ready

## Run on Windows
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Publish
Push the project to GitHub and deploy `app.py` on Streamlit Community Cloud. Choose a memorable app subdomain. Keep sensitive customer data out of a public repository; upload demo/synthetic/de-identified data through the app.

## Judge demo flow
1. Show the animated CustomerIQ landing screen.
2. Upload the dataset.
3. Train the model live.
4. Open **3D Command Center** and explain the customer galaxy.
5. Open **Segment DNA** to show unscaled business profiles.
6. Show **Model Proof** to demonstrate how K was selected.
7. Use **Live Classifier** with a hypothetical customer.
8. Switch themes to demonstrate the polished interactive UX.
