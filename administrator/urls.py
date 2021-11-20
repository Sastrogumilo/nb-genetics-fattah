from django.urls import path, include
from . import views

app_name = 'administrator'

urlpatterns = [
    path('', views.index, name='index'),
    path('dataset/', views.dataset, name='dataset'),
    path('tentang/', views.tentang, name='tentang'),
    path('EDA/', views.EDA, name='EDA'),
    path('encode/', views.encode, name='encode'),
    #path('klasifikasi/', views.klasifikasi, name='klasifikasi'),
    path('NaiveBayes/', views.NaiveBayes, name='NaiveBayes'),
    path('NaiveBayes_Algen/', views.NaiveBayes_Algen, name='NaiveBayes_Algen'),
    path('hasil_nb_algen/', views.hasil_nb_algen, name='hasil_nb_algen'),
    #path('hasilsvmrbf/', views.hasilsvmrbf, name='hasilsvmrbf'),
    #path('SVMRBFIG/', views.SVMRBFIG, name='SVMRBFIG'),


]
