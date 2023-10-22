import numpy as np
import pandas as pd
import streamlit as st
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler

#st.config.set_option('theme.base','auto')

cols=['movement_reactions', 'mentality_composure', 'passing', 'potential',
       'release_clause_eur', 'dribbling', 'wage_eur', 'power_shot_power',
       'value_eur', 'rcm_effective', 'lcm_effective', 'cm_effective',
       'cm_base', 'lcm_base', 'rcm_base', 'mentality_vision',
       'attacking_short_passing', 'physic', 'lam_effective', 'cam_effective',
       'ram_effective', 'cam_base', 'ram_base', 'lam_base',
       'skill_long_passing', 'ls_effective', 'st_effective', 'rs_effective']
model = pickle.load(open('./FifaDataPrediction.pickle', 'rb'))
data=pickle.load(open('./FifaScaler.pickle', 'rb'))
sd=4.361068390461162

scaler=data['sc']
def predict_species(movement_reactions, mentality_composure, passing, potential,
                    release_clause_eur, dribbling, wage_eur, power_shot_power,
                    value_eur, lcm_effective, cm_effective, rcm_effective,
                    lcm_base, cm_base, rcm_base, mentality_vision,
                    attacking_short_passing, physic, lam_effective,
                    ram_effective, cam_effective, lam_base, ram_base,
                    cam_base, skill_long_passing, ls_effective, st_effective,
                    rs_effective):
    input=pd.DataFrame(np.array([[movement_reactions,mentality_composure,passing,potential,
       release_clause_eur,dribbling,wage_eur,power_shot_power,
       value_eur,rcm_effective,lcm_effective,cm_effective,
       cm_base,lcm_base,rcm_base,mentality_vision,
       attacking_short_passing,physic,lam_effective,cam_effective,
       ram_effective,cam_base,ram_base,lam_base,
       skill_long_passing,ls_effective,st_effective,rs_effective]]).astype(np.float64))

    print(input)
    #input=pd.DataFrame(scaler.transform(input),columns=cols)

    dummy_data = pd.DataFrame(data=np.zeros((input.shape[0], len(data['or'].columns))),columns=data['or'].columns)
    dummy_data[cols]=input
    scld=pd.DataFrame(scaler.transform(dummy_data),columns=dummy_data.columns)
    scld=scld[cols]
    #print(scld.columns)
    prediction=model.predict(scld)

    #getting the 95% confidence limit
    ci_upper_bound = prediction + 1.645 * sd
    ci_lower_bound = prediction - 1.645 * sd


    return prediction,ci_lower_bound,ci_upper_bound
def isnum(n):
    try:
        float(n)
    except ValueError:
        return False
    return True
def main():

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Fifa Stats Prediction Model</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    movement_reactions = st.slider("movement_reactions", min_value=0, max_value=100, step=1)
    mentality_composure = st.slider("mentality_composure", min_value=0, max_value=100, step=1)
    passing = st.slider("passing", min_value=0, max_value=100, step=1)
    potential = st.slider("potential", min_value=0, max_value=100, step=1)
    release_clause_eur = st.text_input("release_clause_eur","0")
    dribbling = st.slider("dribbling", min_value=0, max_value=100, step=1)
    wage_eur =  st.text_input("wage_eur", "0")
    power_shot_power = st.slider("power_shot_power", min_value=0, max_value=100, step=1)
    value_eur =  st.text_input("value_eur", "0")
    lcm_effective = st.slider("lcm_effective", min_value=0, max_value=100, step=1)
    cm_effective = st.slider("cm_effective", min_value=0, max_value=100, step=1)
    rcm_effective = st.slider("rcm_effective", min_value=0, max_value=100, step=1)
    lcm_base = st.slider("lcm_base", min_value=0, max_value=100, step=1)
    cm_base = st.slider("cm_base", min_value=0, max_value=100, step=1)
    rcm_base = st.slider("rcm_base", min_value=0, max_value=100, step=1)
    mentality_vision = st.slider("mentality_vision", min_value=0, max_value=100, step=1)
    attacking_short_passing = st.slider("attacking_short_passing", min_value=0, max_value=100, step=1)
    physic = st.slider("physic", min_value=0, max_value=100, step=1)
    lam_effective = st.slider("lam_effective", min_value=0, max_value=100, step=1)
    ram_effective = st.slider("ram_effective", min_value=0, max_value=100, step=1)
    cam_effective = st.slider("cam_effective", min_value=0, max_value=100, step=1)
    lam_base = st.slider("lam_base", min_value=0, max_value=100, step=1)
    ram_base = st.slider("ram_base", min_value=0, max_value=100, step=1)
    cam_base = st.slider("cam_base", min_value=0, max_value=100, step=1)
    skill_long_passing = st.slider("skill_long_passing", min_value=0, max_value=100, step=1)
    ls_effective = st.slider("ls_effective", min_value=0, max_value=100, step=1)
    st_effective = st.slider("st_effective", min_value=0, max_value=100, step=1)
    rs_effective = st.slider("rs_effective", min_value=0, max_value=100, step=1)


    safe_html="""<div style="background-color:#F4D03F;padding:10px;"><h2 style="color:white;text-align:center;">Your flower is safe</h2></div>"""
    danger_html="""<div style="background-color:#F08080;padding:10px;"><h2 style="color:black;text-align:center;">Your flower is poisonous</h2></div>"""

    if st.button("Predict"):
        if isnum(wage_eur) and isnum(release_clause_eur) and isnum(value_eur):
            output=predict_species(movement_reactions, mentality_composure, passing, potential,
                        release_clause_eur, dribbling, wage_eur, power_shot_power,
                        value_eur, lcm_effective, cm_effective, rcm_effective,
                         lcm_base, cm_base, rcm_base, mentality_vision,
                        attacking_short_passing, physic, lam_effective,
                       ram_effective, cam_effective, lam_base, ram_base,
                        cam_base, skill_long_passing, ls_effective, st_effective,
                        rs_effective)
            st.success('The predicted rating is {}'.format(output[0][0]))
            st.success('Confidence interval at 90% confidence: \n')
            st.success('lower limit: {}'.format(output[1][0]))
            st.success('upper limit: {}'.format(output[2][0]))
        else:
            st.error('Enter numeric values only in the textboxes')
if __name__=='__main__':
    main()
