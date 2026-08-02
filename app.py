from __future__ import annotations

from io import BytesIO
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from byte_ndt_physics import (
    ArrayConfig, MaterialConfig, SteeringConfig, GridConfig,
    calculate_array_field, threshold_metrics, maximum_location,
    generic_delay_export, configuration_dict,
)

st.set_page_config(page_title='Byte NDT – Generic Array Designer', page_icon='📡', layout='wide')

language = st.sidebar.radio('Language / Langue', ['English','Français'], horizontal=True)
fr = language == 'Français'

def tr(en, fr_text):
    return fr_text if fr else en
st.image("logo.jpeg", width=160)
st.title(tr('Byte NDT – Generic Ultrasonic Array Designer','Byte NDT – Conception générique de réseaux ultrasonores'))
st.caption(tr('Learn, design, calculate, analyse and export — independently of the hardware manufacturer.', 'Apprendre, concevoir, calculer, analyser et exporter — indépendamment du constructeur.'))
st.info(tr('Generic V1: neutral CSV and JSON outputs. Lecoeur, Mantis and other hardware adapters will be separate plug-ins.', 'V1 générique : sorties neutres CSV et JSON. Lecoeur, Mantis et les autres matériels seront des plug-ins séparés.'))

with st.sidebar:
    st.header('1. Array / Réseau')
    mx = st.slider('Mx',1,32,11); my = st.slider('My',1,32,11)
    lx = st.number_input('Element X (mm)',0.01,50.0,0.15,0.01)
    ly = st.number_input('Element Y (mm)',0.01,50.0,0.15,0.01)
    gx = st.number_input('Gap X (mm)',0.0,20.0,0.05,0.01)
    gy = st.number_input('Gap Y (mm)',0.0,20.0,0.05,0.01)
    windows=['rect','cos','Han','Ham','Blk','tri']
    apox=st.selectbox('Apodization X',windows); apoy=st.selectbox('Apodization Y',windows)

    st.header('2. Materials / Matériaux')
    f = st.number_input('Frequency / Fréquence (MHz)',0.1,25.0,5.0,0.1)
    d1 = st.number_input('Density medium 1 / Densité milieu 1',0.01,30.0,1.0,0.1)
    cp1 = st.number_input('P speed medium 1 / Vitesse P milieu 1 (m/s)',100.0,15000.0,1480.0,10.0)
    d2 = st.number_input('Density solid / Densité solide',0.01,30.0,7.9,0.1)
    cp2 = st.number_input('P speed solid / Vitesse P solide (m/s)',100.0,15000.0,5900.0,10.0)
    cs2 = st.number_input('S speed solid / Vitesse S solide (m/s)',100.0,10000.0,3200.0,10.0)
    wave = st.selectbox('Wave / Onde',['p','s'])

    st.header('3. Steering / Pilotage')
    angt = st.slider('Array mechanical angle / Angle mécanique (°)',-60.0,60.0,10.217,0.1)
    dt0 = st.number_input('Water path DT0 / Hauteur d’eau DT0 (mm)',0.1,500.0,50.8,0.1)
    theta = st.slider('Theta (°)',-80.0,80.0,20.0,0.5)
    phi = st.slider('Phi (°)',-180.0,180.0,0.0,1.0)
    finite = st.toggle('Finite focus / Focalisation finie',False)
    df = st.number_input('Focal distance DF / Distance focale DF (mm)',0.1,1000.0,50.0,1.0) if finite else np.inf

    st.header('4. Field / Champ')
    plane = st.selectbox('Plane / Plan',['XZ','YZ','XY'])
    fixed_name={'XZ':'Fixed Y / Y fixe (mm)','YZ':'Fixed X / X fixe (mm)','XY':'Depth Z / Profondeur Z (mm)'}[plane]
    fixed_min=0.001 if plane=='XY' else -500.0
    fixed_default=10.0 if plane=='XY' else 0.0
    fixed=st.number_input(fixed_name,fixed_min,1000.0 if plane=='XY' else 500.0,fixed_default,1.0)
    hmin=st.number_input('Horizontal min (mm)',-1000.0,1000.0,-5.0,1.0)
    hmax=st.number_input('Horizontal max (mm)',-1000.0,1000.0,20.0,1.0)
    vmin=st.number_input('Vertical min (mm)',-1000.0,1000.0,-20.0 if plane=='XY' else 1.0,1.0)
    vmax=st.number_input('Vertical max (mm)',-1000.0,1000.0,20.0,1.0)
    mode=st.radio('Mode',['Preview / Aperçu','High resolution / Haute résolution'])
    if mode.startswith('Preview'):
        nh=30; nv=30; R=1; Q=1
    else:
        nh=st.slider('Horizontal points',40,180,80,10)
        nv=st.slider('Vertical points',40,180,80,10)
        R=st.slider('Segments X per element',1,8,2)
        Q=st.slider('Segments Y per element',1,8,2)

array=ArrayConfig(mx,my,lx,ly,gx,gy,apox,apoy)
mat=MaterialConfig(f,d1,cp1,d2,cp2,cs2,wave)
steer=SteeringConfig(angt,dt0,theta,phi,df)
grid=GridConfig(plane,hmin,hmax,vmin,vmax,nh,nv,fixed,R,Q)

lambda1=mat.wavelength_1_mm(); lambda2=mat.wavelength_2_mm()
metrics=st.columns(6)
metrics[0].metric('Elements',array.active_elements)
metrics[1].metric('Pitch X',f'{array.pitch_x_mm:.3f} mm')
metrics[2].metric('Pitch Y',f'{array.pitch_y_mm:.3f} mm')
metrics[3].metric('λ medium 1',f'{lambda1:.3f} mm')
metrics[4].metric('λ solid',f'{lambda2:.3f} mm')
metrics[5].metric('Theta / Phi',f'{theta:.1f}° / {phi:.1f}°')

if array.pitch_x_mm/lambda2>0.5 or array.pitch_y_mm/lambda2>0.5:
    st.warning(tr('Pitch exceeds λ/2 in at least one direction. Field analysis is required to assess grating lobes.', 'Le pitch dépasse λ/2 dans au moins une direction. Le calcul du champ est nécessaire pour évaluer les lobes de réseau.'))

tabs=st.tabs(['Learn / Apprendre','Design / Concevoir','Calculate / Calculer','Analyse','Parameter Study / Étude','Export'])

with tabs[0]:
    st.header(tr('2-D array tutorial','Tutoriel réseau 2D'))
    st.markdown(r'''
### 1D versus 2D / Réseau 1D et réseau 2D
A 1-D array mainly controls one angle. A matrix array distributes elements in X and Y and can control **theta** and **phi**.

Un réseau 1D pilote principalement un angle. Une matrice 2D répartit les éléments en X et Y et peut piloter **theta** et **phi**.

### Pitch, gap and frequency / Pitch, gap et fréquence
''')
    st.latex(r"\mathrm{Pitch} = \mathrm{Element\ size} + \mathrm{Gap}")
    st.latex(r"\lambda = \frac{c}{f}")
    st.markdown(r'''
Increasing frequency reduces wavelength. The same physical pitch therefore becomes larger in wavelengths.

L’augmentation de la fréquence réduit la longueur d’onde. Le même pas physique devient donc plus grand lorsqu’il est exprimé en longueurs d’onde.

### Steering and focusing / Pilotage et focalisation
- **theta**: transmitted polar angle in the solid;
- **phi**: azimuth;
- finite `DF`: point focusing;
- `DF = infinity`: steering only.

### V1 assumptions / Hypothèses V1
Planar interface, homogeneous isotropic media, harmonic excitation and rectangular elements. Curved interfaces will be a later module.
''')

with tabs[1]:
    left,right=st.columns(2)
    with left:
        st.subheader('Array / Réseau')
        df_design=pd.DataFrame([
            ['Mx × My',f'{mx} × {my}'],['Active elements',array.active_elements],
            ['Element X',f'{lx:.4f} mm'],['Element Y',f'{ly:.4f} mm'],
            ['Gap X',f'{gx:.4f} mm'],['Gap Y',f'{gy:.4f} mm'],
            ['Pitch X',f'{array.pitch_x_mm:.4f} mm'],['Pitch Y',f'{array.pitch_y_mm:.4f} mm'],
            ['Aperture X',f'{array.aperture_x_mm:.4f} mm'],['Aperture Y',f'{array.aperture_y_mm:.4f} mm'],
            ['Fill factor X',f'{100*array.fill_factor_x:.1f}%'],['Fill factor Y',f'{100*array.fill_factor_y:.1f}%']],columns=['Parameter','Value'])
        st.dataframe(df_design,use_container_width=True,hide_index=True)
    with right:
        st.subheader('Dimensionless indicators / Indicateurs')
        st.dataframe(pd.DataFrame([
            ['λ medium 1',lambda1],['λ selected solid wave',lambda2],
            ['Pitch X / λ solid',array.pitch_x_mm/lambda2],['Pitch Y / λ solid',array.pitch_y_mm/lambda2],
            ['Element X / λ solid',lx/lambda2],['Element Y / λ solid',ly/lambda2]],columns=['Indicator','Value']),use_container_width=True,hide_index=True)

with tabs[2]:
    st.header(tr('Ultrasonic field','Champ ultrasonore'))
    st.caption(tr('High-resolution mode can be slow because every element, segment and field point is evaluated.', 'Le mode haute résolution peut être long car chaque élément, segment et point du champ est calculé.'))
    if st.button(tr('Calculate field','Calculer le champ'),type='primary'):
        if hmax<=hmin or vmax<=vmin:
            st.error(tr('Grid maxima must be greater than minima.','Les maxima de la grille doivent dépasser les minima.'))
        else:
            t0=time.perf_counter()
            try:
                with st.spinner(tr('Calculating...','Calcul en cours...')):
                    result=calculate_array_field(array,mat,steer,grid)
            except Exception as exc:
                st.exception(exc)
            else:
                st.session_state['result']=result
                st.session_state['cfg']=(array,mat,steer,grid)
                st.success(tr(f'Calculation completed in {time.perf_counter()-t0:.2f} s.',f'Calcul terminé en {time.perf_counter()-t0:.2f} s.'))
    if 'result' in st.session_state:
        result=st.session_state['result']
        dyn=st.slider('Dynamic range / Dynamique (dB)',-80,-12,-40,1)
        display=st.radio('Display / Affichage',['dB','Linear / Linéaire'],horizontal=True)
        data=np.maximum(result.magnitude_db,dyn) if display=='dB' else result.magnitude
        fig,ax=plt.subplots(figsize=(10,6))
        im=ax.imshow(data,extent=[result.horizontal_axis_mm.min(),result.horizontal_axis_mm.max(),result.vertical_axis_mm.max(),result.vertical_axis_mm.min()],aspect='auto')
        ax.set_xlabel(result.horizontal_label); ax.set_ylabel(result.vertical_label)
        ax.set_title(tr('Normalized ultrasonic field','Champ ultrasonore normalisé'))
        try:
            ax.contour(result.horizontal_axis_mm,result.vertical_axis_mm,result.magnitude_db,levels=[-12,-6,-3])
        except ValueError:
            pass
        fig.colorbar(im,ax=ax,label='dB' if display=='dB' else tr('Normalized amplitude','Amplitude normalisée'))
        fig.tight_layout(); st.pyplot(fig); st.session_state['figure']=fig

with tabs[3]:
    st.header(tr('Beam-zone analysis','Analyse des zones du faisceau'))
    if 'result' not in st.session_state:
        st.info(tr('Calculate a field first.','Calculez d’abord un champ.'))
    else:
        result=st.session_state['result']
        st.dataframe(pd.DataFrame([threshold_metrics(result,t) for t in (-3.0,-6.0,-12.0)]),use_container_width=True,hide_index=True)
        p=maximum_location(result); c1m,c2m=st.columns(2)
        c1m.metric(f'Maximum {result.horizontal_label}',f"{p['horizontal_mm']:.3f} mm")
        c2m.metric(f'Maximum {result.vertical_label}',f"{p['vertical_mm']:.3f} mm")
        st.write(tr('These 2-D descriptors prepare the later 3-D −3/−6/−12 dB volumes, oriented bounding box and indication-response module.', 'Ces descripteurs 2D préparent les futurs volumes 3D −3/−6/−12 dB, le parallélépipède orienté et le module de réponse sur indication.'))

with tabs[4]:
    st.header(tr('Frequency–pitch study','Étude fréquence–pitch'))
    freqs=np.linspace(max(0.1,f*0.4),f*1.6,80)
    wav=mat.c2_m_s/(1000*freqs)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(freqs,array.pitch_x_mm/wav,label='Pitch X / λ')
    ax.plot(freqs,array.pitch_y_mm/wav,label='Pitch Y / λ')
    ax.axhline(0.5,linestyle='--',label='λ/2 reference')
    ax.set_xlabel(tr('Frequency (MHz)','Fréquence (MHz)')); ax.set_ylabel('Pitch / λ')
    ax.set_title(tr('Pitch expressed in solid wavelengths','Pitch exprimé en longueurs d’onde dans le solide'))
    ax.grid(True); ax.legend(); fig.tight_layout(); st.pyplot(fig)
    st.write(tr('This relation links frequency, wavelength, pitch and gap. The field calculation is then used to observe the actual beam and lobes.', 'Cette relation lie fréquence, longueur d’onde, pitch et gap. Le calcul du champ sert ensuite à observer le faisceau réel et les lobes.'))

with tabs[5]:
    st.header(tr('Generic Byte NDT exports','Exports génériques Byte NDT'))
    config_json=json.dumps(configuration_dict(array,mat,steer,grid),indent=2,ensure_ascii=False)
    st.download_button(tr('Download configuration JSON','Télécharger la configuration JSON'),config_json.encode('utf-8'),'byte_ndt_generic_array_configuration.json','application/json',use_container_width=True)
    if 'result' not in st.session_state:
        st.info(tr('Field and focal-law exports appear after calculation.','Les exports du champ et des lois apparaissent après le calcul.'))
    else:
        result=st.session_state['result']; ca,cm,cs,cg=st.session_state['cfg']
        laws=pd.DataFrame(generic_delay_export(ca,cm,cs,result))
        st.dataframe(laws,use_container_width=True,hide_index=True)
        st.download_button(tr('Download focal laws CSV','Télécharger les lois focales CSV'),laws.to_csv(index=False).encode('utf-8'),'byte_ndt_generic_focal_laws.csv','text/csv',use_container_width=True)
        H,V=np.meshgrid(result.horizontal_axis_mm,result.vertical_axis_mm)
        fields=pd.DataFrame({result.horizontal_label:H.ravel(),result.vertical_label:V.ravel(),'magnitude_normalized':result.magnitude.ravel(),'magnitude_db':result.magnitude_db.ravel(),'vx_real':np.real(result.vx).ravel(),'vx_imag':np.imag(result.vx).ravel(),'vy_real':np.real(result.vy).ravel(),'vy_imag':np.imag(result.vy).ravel(),'vz_real':np.real(result.vz).ravel(),'vz_imag':np.imag(result.vz).ravel()})
        st.download_button(tr('Download field CSV','Télécharger le champ CSV'),fields.to_csv(index=False).encode('utf-8'),'byte_ndt_generic_ultrasonic_field.csv','text/csv',use_container_width=True)
        if 'figure' in st.session_state:
            buf=BytesIO(); st.session_state['figure'].savefig(buf,format='png',dpi=220,bbox_inches='tight')
            st.download_button(tr('Download field PNG','Télécharger l’image PNG'),buf.getvalue(),'byte_ndt_generic_ultrasonic_field.png','image/png',use_container_width=True)
    st.warning(tr('Generic exports must not be loaded into real hardware before MATLAB/Python validation, element-order verification and a verified manufacturer adapter.', 'Les exports génériques ne doivent pas être chargés dans un appareil réel avant validation MATLAB/Python, contrôle de l’ordre des éléments et utilisation d’un adaptateur constructeur vérifié.'))

st.divider()
st.caption('Byte NDT Generic Array App V1 · Planar interface · Generic CSV/JSON · Hardware-independent core')
