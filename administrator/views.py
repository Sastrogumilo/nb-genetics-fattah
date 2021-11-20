from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.utils.decorators import decorator_from_middleware
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.contrib import messages
import random
from operator import itemgetter
import zipfile
import os
from shutil import copyfile

#Plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import plot


#End Plotly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from time import time

#Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn import naive_bayes
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from genetic_selection import GeneticSelectionCV


import time
import numpy as np


# Create your views here.

@login_required(login_url=settings.LOGIN_URL)
def index(request):
    return render(request, 'administrator/dashboard.html')

@login_required(login_url=settings.LOGIN_URL)
def tentang(request):
    return render(request, 'administrator/tentang.html')

#@login_required(login_url=settings.LOGIN_URL)
#def SVM(request):
#    return render(request, 'administrator/SVM.html')

@login_required(login_url=settings.LOGIN_URL)
def NaiveBayes_Algen(request):
    return render(request, 'administrator/NaiveBayes_Algen.html')

#@login_required(login_url=settings.LOGIN_URL)
#def SVMRBFIG(request):
#    return render(request, 'administrator/SVMRBFIG.html')


##Main Module

@login_required(login_url=settings.LOGIN_URL)
def dataset(request):
    if request.method == 'POST':
        file = request.FILES['data']
        if default_storage.exists('dataset.csv'):
            default_storage.delete('dataset.csv')
        file_name = default_storage.save('dataset.csv', file)

        dataset = []
        data = pd.read_csv(default_storage.path('dataset.csv'))
        data.rename(columns={   
                            'gender':'Gender',
                            'NationalITy':'Nationality',
                            'VisITedResources':'VisitedResources',
                            'ParentschoolSatisfaction':'ParentSchoolSatisfaction',
                            'raisedhands':'raisedHands'
                            }, inplace=True) 
        
        data['Nationality'].replace({
                                    'KW':'Kuwait',
                                    'venzuela':'Venezuela',
                                    'Lybia':'Libya',
                                    'lebanon':'Lebanon',
                                    'Tunis':'Tunisia'
                                    },inplace=True)
        
        data['PlaceofBirth'].replace({
                                    'venzuela':'Venezuela',
                                    'Lybia':'Libya',
                                    'lebanon':'Lebanon',
                                    'KuwaIT':'Kuwait',
                                    'Tunis':'Tunisia'
                                    },inplace=True)
        data.to_csv("media/dataset.csv", index=False)
        
        for x in range(len(data['Gender'])):
            temp = []
            temp.append(data['Gender'][x])
            temp.append(data['Nationality'][x])
            temp.append(data['PlaceofBirth'][x])
            temp.append(data['StageID'][x])
            temp.append(data['GradeID'][x])
            temp.append(data['SectionID'][x])
            temp.append(data['Topic'][x])
            temp.append(data['Semester'][x])
            temp.append(data['Relation'][x])
            temp.append(data['raisedHands'][x])
            temp.append(data['VisitedResources'][x])
            temp.append(data['AnnouncementsView'][x])
            temp.append(data['Discussion'][x])
            temp.append(data['ParentAnsweringSurvey'][x])
            temp.append(data['ParentSchoolSatisfaction'][x])
            temp.append(data['StudentAbsenceDays'][x])
            temp.append(data['Class'][x])
            
            # print(data['sentiment'][x])
            dataset.append(temp)
        # path = default_storage.save('dataset.csv', ContentFile(file.read()))
        messages.success(request,'Dataset berhasil diupload!')

        return render(request, 'administrator/dataset.html',{'dataset': dataset})
    else:
        if default_storage.exists('dataset.csv'):
            dataset = []
            data = pd.read_csv(default_storage.path('dataset.csv'))
            
            data.rename(columns={   
                            'gender':'Gender',
                            'NationalITy':'Nationality',
                            'VisITedResources':'VisitedResources',
                            'ParentschoolSatisfaction':'ParentSchoolSatisfaction',
                            'raisedhands':'raisedHands'
                            }, inplace=True) 
        
        data['Nationality'].replace({
                                    'KW':'Kuwait',
                                    'venzuela':'Venezuela',
                                    'Lybia':'Libya',
                                    'lebanon':'Lebanon',
                                    'Tunis':'Tunisia'
                                    },inplace=True)
        
        data['PlaceofBirth'].replace({
                                    'venzuela':'Venezuela',
                                    'Lybia':'Libya',
                                    'lebanon':'Lebanon',
                                    'KuwaIT':'Kuwait',
                                    'Tunis':'Tunisia'
                                    },inplace=True)
        
        data.to_csv("media/dataset.csv", index=False)    
        print(data['Class'])
        for x in range(len(data['Gender'])):
            temp = []
            temp.append(data['Gender'][x])
            temp.append(data['Nationality'][x])
            temp.append(data['PlaceofBirth'][x])
            temp.append(data['StageID'][x])
            temp.append(data['GradeID'][x])
            temp.append(data['SectionID'][x])
            temp.append(data['Topic'][x])
            temp.append(data['Semester'][x])
            temp.append(data['Relation'][x])
            temp.append(data['raisedHands'][x])
            temp.append(data['VisitedResources'][x])
            temp.append(data['AnnouncementsView'][x])
            temp.append(data['Discussion'][x])
            temp.append(data['ParentAnsweringSurvey'][x])
            temp.append(data['ParentSchoolSatisfaction'][x])
            temp.append(data['StudentAbsenceDays'][x])
            temp.append(data['Class'][x])
            
            # print(data['sentiment'][x])
            dataset.append(temp)

            # with open(default_storage.path('dataset.csv'), 'r') as data:
            #     reader = csv.reader(data)
            #     dataset = []
            #     for row in reader:
            #         dataset.append(row)
            # print(dataset)

        else:
            dataset = []
        # nama=[]
        # jumlah=[]
        # dataset=[]
        # if default_storage.exists('dataset'):
        #     for name in os.listdir(os.path.join(settings.BASE_DIR, 'media/dataset')):
        #         dataset.append([str(name),str(len(os.listdir(os.path.join(settings.BASE_DIR, 'media/dataset/'+name))))])
        # # print(dataset)
        
        return render(request, 'administrator/dataset.html',{'dataset': dataset})



@login_required(login_url=settings.LOGIN_URL)
def EDA(request):
    
    if default_storage.exists('dataset.csv'):
        data = pd.read_csv(default_storage.path('dataset.csv'))
        
        #Graf Kelas
        kelas_data = px.bar(
                            data, x = data['Class'].unique(), y = data['Class'].value_counts(), 
                            color = data['Class'].unique(), 
                            title="Graf Kelas",
                            labels={
                                "x": "Kelas",
                                "y": "Jumlah"
                                }
                            )
        plot_kelas = plot(kelas_data, output_type='div')
        
        #Graf Gender
        gender_data = px.bar(
                            data, x = data['Gender'].unique(), y = data['Gender'].value_counts(), 
                            color = data['Gender'].unique(), 
                            title="Graf Gender",
                            labels={
                                "x": "Gender",
                                "y": "Jumlah"
                                }
                            )
        plot_gender = plot(gender_data, output_type='div')
        
        #Graf StageID
        stageid_data = px.bar(
                            data, x = data['StageID'].unique(), y = data['StageID'].value_counts(), 
                            color = data['StageID'].unique(), 
                            title="Graf Jenjang Sekolah",
                            labels={
                                "x": "Jenjang Sekolah",
                                "y": "Jumlah"
                                }
                            )
        plot_stageid = plot(stageid_data, output_type='div')
        
        #Graf Semester
        semester_data = px.bar(
                            data, x = data['Semester'].unique(), y = data['Semester'].value_counts(),
                            color = data['Semester'].unique(),
                            title="Graf Semester",
                            labels={
                                "x": "Semester",
                                "y": "Jumlah"
                                }
                            )
        plot_semester = plot(semester_data, output_type='div')
        
        #Graf Topic
        topic_data = px.bar(
                            data, x = data['Topic'].unique(), y = data['Topic'].value_counts(), 
                            color = data['Topic'].unique(), 
                            title="Graf Topic",
                            labels={
                                "x": "Topic",
                                "y": "Jumlah"
                                }
                            )
        plot_topic = plot(topic_data, output_type='div')
        
        #Graf Nationality
        nationality_data = px.bar(
                            data, x = data['Nationality'].unique(), y = data['Nationality'].value_counts(), 
                            color = data['Nationality'].unique(), 
                            title="Graf Kebangsaan",
                            labels={
                                "x": "Kebangsaan",
                                "y": "Jumlah"
                                }
                            )
        plot_nationality = plot(nationality_data, output_type='div')
        
        #Visualisasi Fitur "Gender" dalam fitur "Class","Relation","StudentAbsenceDays","ParentAnsweringSurvey"
        gender_to_class_data = px.histogram(
                            data, x = 'Gender', y = 'Class',
                            barmode='group',
                            color = 'Class', 
                            title="Graf Fitur Jenis Kelamin Dalam Fitur Kelas ",
                            labels={
                                "Gender": "Jumlah Gender",
                                "Class": "Kelas"
                                },
                            )
        plot_gender_to_class = plot(gender_to_class_data, output_type='div')
        
        gender_to_relation_data = px.histogram(
                            data, x = 'Gender', y = 'Relation',
                            barmode='group',
                            color = 'Relation', 
                            title="Graf Fitur Jenis Kelamin Dalam Fitur Relasi Orang Tua ",
                            labels={
                                "Gender": "Jenis Kelamin",
                                "Relation": "Relasi"
                                },
                            )
        plot_gender_to_relation = plot(gender_to_relation_data, output_type='div')
        
        
        gender_to_absen_data = px.histogram(
                            data, x = 'Gender', y = 'StudentAbsenceDays',
                            barmode='group',
                            color = 'StudentAbsenceDays', 
                            title="Graf Fitur Jenis Kelamin Dalam Fitur Absen Siswa ",
                            labels={
                                "Gender": "Jenis Kelamin",
                                "Relation": "Relasi"
                                },
                            )
        plot_gender_to_absen = plot(gender_to_absen_data, output_type='div')
        
        
        gender_to_parentSurvey_data = px.histogram(
                            data, x = 'Gender', y = 'ParentAnsweringSurvey',
                            barmode='group',
                            color = 'ParentAnsweringSurvey', 
                            title="Graf Fitur Jenis Kelamin Dalam Fitur Survey Orang Tua ",
                            labels={
                                "Gender": "Jenis Kelamin",
                                "Relation": "Relasi"
                                },
                            )
        plot_gender_to_parentSurvey = plot(gender_to_parentSurvey_data, output_type='div')
        
        
        gender_to_topic_data = px.histogram(
                            data, x = 'Gender', y = 'Topic',
                            barmode='group',
                            color = 'Topic', 
                            title="Graf Fitur Jenis Kelamin Dalam Fitur Survey Orang Tua ",
                            labels={
                                "Gender": "Jenis Kelamin",
                                "Relation": "Relasi"
                                },
                            )
        plot_gender_to_topic = plot(gender_to_topic_data, output_type='div')
        
        gender_to_nationality_data = px.histogram(
                            data, x = 'Gender', y = 'Nationality',
                            barmode='group',
                            color = 'Nationality', 
                            title="Graf Fitur Jenis Kelamin Dalam Fitur Kebangsaan ",
                            labels={
                                "Gender": "Jenis Kelamin",
                                "Relation": "Relasi"
                                },
                            )
        plot_gender_to_nationality = plot(gender_to_nationality_data, output_type='div')
        
    
        return render(request, 'administrator/EDA.html',context={
                                                                'plot_div_kelas': plot_kelas,
                                                                'plot_div_gender': plot_gender,
                                                                'plot_div_stageid': plot_stageid,
                                                                'plot_div_semester': plot_semester,
                                                                'plot_div_topic': plot_topic,
                                                                'plot_div_nationality': plot_nationality,
                                                                'plot_div_gender_to_class': plot_gender_to_class,
                                                                'plot_div_gender_to_relation': plot_gender_to_relation,
                                                                'plot_div_gender_to_absen': plot_gender_to_absen,
                                                                'plot_div_gender_to_parentSurvey': plot_gender_to_parentSurvey,
                                                                'plot_div_gender_to_topic': plot_gender_to_topic,
                                                                'plot_div_gender_to_nationality': plot_gender_to_nationality,
                                                                                                                        
                                                                }
                    
                    )
    else:
        messages.error(request,'Dataset belum diinputkan!')
        return redirect('/administrator/dataset/')
        pass

@login_required(login_url=settings.LOGIN_URL)

def encode(request):
    if default_storage.exists('dataset.csv'):
        
        data = pd.read_csv(default_storage.path('dataset.csv'))
        Features = data.drop('Gender',axis=1)
        Target = data['Gender']
        label = LabelEncoder()
        Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
        for col in Cat_Colums:
            Features[col] = label.fit_transform(Features[col])

        #Semester Encoding
        Features = data.drop('Semester',axis=1)
        Target = data['Semester']
        label = LabelEncoder()
        Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
        for col in Cat_Colums:
            Features[col] = label.fit_transform(Features[col])

        #Parent Survey Encoding
        Features = data.drop('ParentAnsweringSurvey',axis=1)
        Target = data['ParentAnsweringSurvey']
        label = LabelEncoder()
        Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
        for col in Cat_Colums:
            Features[col] = label.fit_transform(Features[col])

        #Relational Encoding
        Features = data.drop('Relation',axis=1)
        Target = data['Relation']
        label = LabelEncoder()
        Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
        for col in Cat_Colums:
            Features[col] = label.fit_transform(Features[col])

        #Parent Saticfactional
        Features = data.drop('ParentSchoolSatisfaction',axis=1)
        Target = data['ParentSchoolSatisfaction']
        label = LabelEncoder()
        Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
        for col in Cat_Colums:
            Features[col] = label.fit_transform(Features[col])

        #Students Absence
        Features = data.drop('StudentAbsenceDays',axis=1)
        Target = data['StudentAbsenceDays']
        label = LabelEncoder()
        Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
        for col in Cat_Colums:
            Features[col] = label.fit_transform(Features[col])

        #Class Encoding
        Features = data.drop('Class',axis=1)
        Target = data['Class']
        label = LabelEncoder()
        Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
        for col in Cat_Colums:
            Features[col] = label.fit_transform(Features[col])
            
        Features.to_csv('media/fitur.csv', index=False)
        Target.to_csv('media/target.csv', index=False, header=True)
        
        data_fitur = []
        fitur = pd.read_csv(default_storage.path('fitur.csv'))
        for x in range(len(fitur['Gender'])):
            temp = []
            temp.append(fitur['Gender'][x])
            temp.append(fitur['Nationality'][x])
            temp.append(fitur['PlaceofBirth'][x])
            temp.append(fitur['StageID'][x])
            temp.append(fitur['GradeID'][x])
            temp.append(fitur['SectionID'][x])
            temp.append(fitur['Topic'][x])
            temp.append(fitur['Semester'][x])
            temp.append(fitur['Relation'][x])
            temp.append(fitur['raisedHands'][x])
            temp.append(fitur['VisitedResources'][x])
            temp.append(fitur['AnnouncementsView'][x])
            temp.append(fitur['Discussion'][x])
            temp.append(fitur['ParentAnsweringSurvey'][x])
            temp.append(fitur['ParentSchoolSatisfaction'][x])
            temp.append(fitur['StudentAbsenceDays'][x])
            
            # print(data['sentiment'][x])
            data_fitur.append(temp)
            
        data_target = []
        target = pd.read_csv(default_storage.path('target.csv'))
        for x in range(len(target['Class'])):
            temp = []
            temp.append(target['Class'][x])
            data_target.append(temp)
            
        return render(request, 'administrator/encode.html',{'DataFitur': data_fitur, 'DataTarget': data_target})
    else:
        messages.error(request, 'Dataset belum diinputkan!')
        return redirect('/administrator/dataset/')
        pass

@login_required(login_url=settings.LOGIN_URL)
def NaiveBayes(request):
    if default_storage.exists('dataset.csv'):
        Target = pd.read_csv(default_storage.path('target.csv'))
        Features = pd.read_csv(default_storage.path('fitur.csv'))
        X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=0.3, random_state=52)
        
        print(X_test)
        print(y_test)
        print(X_train.columns)
        
        Logit_Model = naive_bayes.GaussianNB()
        Logit_Model.fit(X_train,y_train)
        Prediction = Logit_Model.predict(X_test)
        Score = accuracy_score(y_test,Prediction)
        Report = classification_report(y_test,Prediction)

        print(Prediction)
        print("Accuracy Score: {}%".format(Score*100))
        print(Report)
        
        labels=['High', 'Low', 'Mid']
        preds = np.array(Logit_Model.predict(X_test))
        #preds2 = Logit_Model.score(X_test, y_test)
        #preds = np.argmax(preds, axis = -1)
        orig = y_test
        conf = confusion_matrix(orig, preds)
        
        fig = ff.create_annotated_heatmap(conf, colorscale='blues', x=labels, y=labels)
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
                        xaxis_title="Predicted",
                        yaxis_title="Truth",
                        xaxis = dict(
                                        tickmode = 'array',
                                        tickvals = [0,1,2],
                                        ticktext = labels
                                    ),
                        yaxis = dict(
                                        tickmode = 'array',
                                        tickvals = [0,1,2],
                                        ticktext = labels
                                    ),
                        autosize = False,
                        width = 600,
                        height = 600,
                        )
                        
        fig.update_xaxes(side="top")
        plot_conf_nb = plot(fig, output_type='div')
        
        csv_report_nb = pd.DataFrame(classification_report(orig,
                                                    preds,
                                                    output_dict=True,
                                                    
                                                    )
                                    )
        
        csv_report_nb = csv_report_nb.iloc[:-1, :].T
        csv_report_nb = csv_report_nb.round(3)
        
        z = csv_report_nb.values.tolist()
        x = csv_report_nb.columns.to_list()
        y = csv_report_nb.index.tolist()


        fig_report = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='blues', 
                                        #x=['High', 'Low', 'Mid'], y=['High', 'Low', 'Mid']
                                        )

        #fig.update_yaxes(autorange="reversed")
        fig_report.update_layout(title="Plot Title",
                        #xaxis_title="x Axis Title",
                        #yaxis_title="y Axis Title",
                        autosize = False,
                        width = 700,
                        height = 700,                  
                        )
        
        for i in range(len(fig_report.layout.annotations)):
                fig_report.layout.annotations[i].font.size = 15
                        
        fig_report.update_yaxes(categoryorder='category descending')
        plot_report_nb = plot(fig_report, output_type='div')

        
        return render(request, 'administrator/NaiveBayes.html',{'Report': plot_report_nb, 'skor_acc':Score, 'plot_div_conf_nb': plot_conf_nb })

    else:
        messages.error(request,'Dataset belum diinputkan!')
        return redirect('/administrator/dataset/')
        pass


@login_required(login_url=settings.LOGIN_URL)
def hasil_nb_algen(request):
    if request.method == 'GET':
        jumlah_fitur = int(request.GET['jumlah_fitur'])
        jumlah_populasi = int(request.GET['jumlah_populasi'])
        crossover = float(request.GET['crossover'])
        mutasi = float(request.GET['mutasi'])
        jumlah_generasi = int(request.GET['jumlah_generasi'])
        jumlah_gen_no_change = int(request.GET['jumlah_gen_no_change'])
        
        rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=69420213)
        
        estimator = naive_bayes.GaussianNB()
        
        Target = pd.read_csv(default_storage.path('target.csv'))
        Features = pd.read_csv(default_storage.path('fitur.csv'))
        X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=0.3, random_state=52)
        
        selector = GeneticSelectionCV(estimator,
                                  cv=rkf, #Cross Validation
                                  verbose=1,
                                  scoring="accuracy",
                                  max_features=jumlah_fitur, #jumlah fitur
                                  n_population=jumlah_populasi,
                                  crossover_proba=crossover,
                                  mutation_proba=mutasi,
                                  n_generations=jumlah_generasi,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  #tournament_size=3,
                                  n_gen_no_change=jumlah_gen_no_change,
                                  caching=True,
                                  n_jobs=-1)
        
        selector = selector.fit(X_train,y_train)
        
        genfeats = X_train.columns[selector.support_]
        genfeats = list(genfeats)
        
        y_pred = selector.predict(X_test)
        
        val_akurasi = accuracy_score(y_test,y_pred)
        
        train_akurasi = selector.generation_scores_[-1]
        
        labels=['High', 'Low', 'Mid']
        preds = np.array(selector.predict(X_test))
        #preds2 = Logit_Model.score(X_test, y_test)
        #preds = np.argmax(preds, axis = -1)
        orig = y_test
        conf = confusion_matrix(orig, preds)
        
        fig = ff.create_annotated_heatmap(conf, colorscale='blues', x=labels, y=labels)
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
                        xaxis_title="Predicted",
                        yaxis_title="Truth",
                        xaxis = dict(
                                        tickmode = 'array',
                                        tickvals = [0,1,2],
                                        ticktext = labels
                                    ),
                        yaxis = dict(
                                        tickmode = 'array',
                                        tickvals = [0,1,2],
                                        ticktext = labels
                                    ),
                        autosize = False,
                        width = 600,
                        height = 600,
                        )
                        
        fig.update_xaxes(side="top")
        plot_conf_nb_algen = plot(fig, output_type='div')
        
        csv_report_nb = pd.DataFrame(classification_report(orig,
                                                    preds,
                                                    output_dict=True,
                                                    
                                                    )
                                    )
        
        csv_report_nb = csv_report_nb.iloc[:-1, :].T
        csv_report_nb = csv_report_nb.round(3)
        
        z = csv_report_nb.values.tolist()
        x = csv_report_nb.columns.to_list()
        y = csv_report_nb.index.tolist()


        fig_report = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='blues', 
                                        #x=['High', 'Low', 'Mid'], y=['High', 'Low', 'Mid']
                                        )

        #fig.update_yaxes(autorange="reversed")
        fig_report.update_layout(title="Plot Title",
                        #xaxis_title="x Axis Title",
                        #yaxis_title="y Axis Title",
                        autosize = False,
                        width = 700,
                        height = 700,                  
                        )
        
        for i in range(len(fig_report.layout.annotations)):
                fig_report.layout.annotations[i].font.size = 15
                        
        fig_report.update_yaxes(categoryorder='category descending')
        plot_report_nb_algen = plot(fig_report, output_type='div')
        

        return render(request, 'administrator/hasil_nb_algen.html',{'fitur_terpilih':genfeats, 
                                                                 'val_akurasi': val_akurasi, 
                                                                 'train_akurasi': train_akurasi,
                                                                 'plot_div_conf_nb_algen': plot_conf_nb_algen,
                                                                 'plot_div_report_nb_algen': plot_report_nb_algen,
                                                                 })

    else:
        messages.error(request,'Dataset belum diinputkan!')
        return redirect('/administrator/dataset/')
        pass
