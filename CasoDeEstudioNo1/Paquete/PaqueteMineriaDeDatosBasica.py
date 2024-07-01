import pandas as pd
import numpy as np
import umap as um
import math
import statistics
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets, ensemble
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil, floor, pi
from prince import PCA as PCA_Prince
from seaborn import color_palette
from scipy.cluster.hierarchy import dendrogram, ward, single, complete,average,linkage, fcluster
from sklearn.decomposition import PCA
pd.options.display.max_rows = 10
import warnings
warnings.filterwarnings('ignore')

class AnalisisDatosExploratorio():
    def __init__(self, path, num):
        self.__df = self.__cargarDatos(path, num)  
    @property
    def df(self):
        return self.__df 
    @df.setter
    def df(self, p_df):
        self.__df = p_df
            
    def analisisNumerico(self):
        self.__df = self.__df.select_dtypes(include = ["number"])

    def analisisCompleto(self):  
        self.__df = pd.get_dummies(self.__df)
            
    def __cargarDatos(self, path, num):
        if num == 1:
            return pd.read_csv(path,
            sep = ",",
            decimal = ".",
            index_col = 0)
        if num == 2:
            return pd.read_csv(path,
            sep = ";",
            decimal = ".")
     
    def analisis(self):
        self.__df = pd.DataFrame(StandardScaler().fit_transform(self.__df),columns=self.__df.columns,index = self.__df.index)
        print("Dimensiones:",self.__df.shape)
        print(self.__df.head)
        print(self.__df.describe())
        self.__df.dropna().describe()
        self.__df.mean(numeric_only=True)
        self.__df.median(numeric_only=True)
        self.__df.std(numeric_only=True, ddof = 0) 
        self.__df.max(numeric_only=True)
        self.__df.min(numeric_only=True)
        self.__df.quantile(np.array([0,.33,.50,.75,1]),numeric_only=True)
        self.__graficosBoxplot()
        self.__funcionDensidad()
        self.__histograma()
        self.__correlaciones()
        self.__graficoDeCorrelacion()
   
    def __graficosBoxplot(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (15,8), dpi = 200)
        boxplots = self.__df.boxplot(return_type='axes',ax=ax)
        plt.show()
          
    def __funcionDensidad(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (12,8), dpi = 200)
        densidad = self.__df[self.__df.columns].plot(kind='density',ax = ax)
        plt.show()
                 
    def __histograma(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10,6), dpi = 200)
        densidad = self.__df[self.__df.columns].plot(kind='hist', ax = ax)
        plt.show()
      
    def __correlaciones(self):
        corr = self.__df.corr(numeric_only=True)
        print(corr)

    def __graficoDeCorrelacion(self):
        fig, ax = plt.subplots(figsize=(12, 8), dpi = 150)
        paleta = sns.diverging_palette(220, 10,as_cmap=True).reversed()
        corr = self.__df.corr(numeric_only=True)
        sns.heatmap(corr, vmin= -1, vmax=1, cmap= paleta,square=True, annot=True, ax=ax)
        plt.show()

    def centroide(num_cluster, datos, clusters):
        ind = clusters == num_cluster
        return(pd.DataFrame(datos[ind].mean()).T)

    def recodificar(col, nuevo_codigo):
        col_cod = pd.Series(col, copy=True)
        for llave, valor in nuevo_codigo.items():
            col_cod.replace(llave, valor, inplace=True)
        return col_cod

    def bar_plot(centros, labels, scale = False,cluster = None, var = None):
        fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 200)
        centros = np.copy(centros)
        if scale:
            for col in range(centros.shape[1]):
                centros[:,col] = centros[:,col] / max(centros[:,col])
        colores = color_palette()
        minimo = floor(centros.min()) if floor(centros.min()) < 0 else 0
        def inside_plot(valores, labels, titulo):
            plt.barh(range(len(valores)), valores, 1/1.5, color = colores)
            plt.xlim(minimo, ceil(centros.max()))
            plt.title(titulo)
        if var is not None:
            centros = np.array([n[[x in var for x in labels]] for n in centros])
            colores = [colores[x % len(colores)] for x, i in enumerate(labels) if i in var]
            labels = labels[[x in var for x in labels]]
        if cluster is None:
            for i in range(centros.shape[0]):
                plt.subplot(1, centros.shape[0], i + 1)
                inside_plot(centros[i].tolist(), labels, ('Cluster ' + str(i)))
                plt.yticks(range(len(labels)), labels) if i == 0 else plt.yticks([]) 
        else:
            pos = 1
            for i in cluster:
                plt.subplot(1, len(cluster), pos)
                inside_plot(centros[i].tolist(), labels, ('Cluster ' + str(i)))
                plt.yticks(range(len(labels)), labels) if pos == 1 else plt.yticks([]) 
                pos += 1
                
    def bar_plot_detail(centros,columns_names = [], columns_to_plot = [],figsize = (10,7),dpi = 150):
        fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 200)
        numClusters = centros.shape[0]
        labels = ["Cluster "+ str(i) for i in range(numClusters)]
        centros = pd.DataFrame(centros,columns=columns_names,index= labels)
        plots = len(columns_to_plot) if len(columns_to_plot) != 0 else len(columns_names)
        rows, cols = ceil(plots/2),2
        plt.figure(1, figsize = figsize,dpi = dpi)
        plt.subplots_adjust(hspace=1,wspace = 0.5)
        columns = columns_names
        if len(columns_to_plot) > 0: 
            if type(columns_to_plot[0]) is str:
                columns = columns_to_plot
            else:
                columns = [columns_names[i] for i in columns_to_plot]
        var = 0
        for numRow in range(rows):
            for numCol in range(cols):
                if var < plots:
                    ax = plt.subplot2grid((rows, cols), (numRow, numCol), colspan=1, rowspan=1)
                    sns.barplot(y = labels, x=columns[var] ,data=centros ,ax=ax)
                    var += 1    
                
    def radar_plot(centros, labels):
        fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 200)
        centros = np.array([((n - min(n)) / (max(n) - min(n)) * 100) if 
                            max(n) != min(n) else (n/n * 50) for n in centros.T])
        angulos = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
        angulos += angulos[:1]
        ax = plt.subplot(111, polar = True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angulos[:-1], labels)
        ax.set_rlabel_position(0)
        plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"], 
            color = "grey", size = 8)
        plt.ylim(-10, 100)
        for i in range(centros.shape[1]):
            valores = centros[:, i].tolist()
            valores += valores[:1]
            ax.plot(angulos, valores, linewidth = 1, linestyle = 'solid', 
                    label = 'Cluster ' + str(i))
            ax.fill(angulos, valores, alpha = 0.3)
        plt.legend(loc='upper right', bbox_to_anchor = (0.1, 0.1))

    def __str__(self):
        return f'AnalisisDatosExploratorio: {self.__df}'

class ACPBasico:
    def __init__(self, datos, n_componentes = 2): 
        self.__datos = datos
        self.__modelo = PCA_Prince(n_components = n_componentes).fit(self.__datos)
        self.__correlacion_var = self.__modelo.column_correlations
        self.__coordenadas_ind = self.__modelo.row_coordinates(self.__datos)
        self.__contribucion_ind = self.__modelo.row_contributions_
        self.__cos2_ind = self.__modelo.row_cosine_similarities(self.__datos)
        self.__var_explicada = self.__modelo.percentage_of_variance_
    @property
    def datos(self):
        return self.__datos
    @datos.setter
    def datos(self, datos):
        self.__datos = datos
    @property
    def modelo(self):
        return self.__modelo
    @property
    def correlacion_var(self):
        return self.__correlacion_var
    @property
    def coordenadas_ind(self):
        return self.__coordenadas_ind
    @property
    def contribucion_ind(self):
        return self.__contribucion_ind
    @property
    def cos2_ind(self):
        return self.__cos2_ind
    @property
    def var_explicada(self):
        return self.__var_explicada
    @var_explicada.setter
    def var_explicada(self, var_explicada):
        self.__var_explicada = var_explicada
    @modelo.setter
    def modelo(self, modelo):
        self.__modelo = modelo
    @correlacion_var.setter
    def correlacion_var(self, correlacion_var):
        self.__correlacion_var = correlacion_var
    @coordenadas_ind.setter
    def coordenadas_ind(self, coordenadas_ind):
        self.__coordenadas_ind = coordenadas_ind
    @contribucion_ind.setter
    def contribucion_ind(self, contribucion_ind):
        self.__contribucion_ind = contribucion_ind
    @cos2_ind.setter
    def cos2_ind(self, cos2_ind):
        self.__cos2_ind = cos2_ind

    def plot_plano_principal(self, ejes = [0, 1], ind_labels = True, titulo = 'Plano Principal'):
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.scatter(x, y, color = 'gray')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))

    def plot_circulo(self, ejes = [0, 1], var_labels = True, titulo = 'Círculo de Correlación'):
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-v0_8-whitegrid')
        c = plt.Circle((0, 0), radius = 1, color = 'steelblue', fill = False)
        plt.gca().add_patch(c)
        plt.axis('scaled')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * 0.95, cor[i, 1] * 0.95, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * 1.05, cor[i, 1] * 1.05, self.correlacion_var.index[i], 
                         color = 'steelblue', ha = 'center', va = 'center')

    def plot_sobreposicion(self, ejes = [0, 1], ind_labels = True, 
                      var_labels = True, titulo = 'Sobreposición Plano-Círculo'):
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        cor = self.correlacion_var.iloc[:, ejes]
        scale = min((max(x) - min(x)/(max(cor[ejes[0]]) - min(cor[ejes[0]]))), 
                    (max(y) - min(y)/(max(cor[ejes[1]]) - min(cor[ejes[1]])))) * 0.7
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        plt.scatter(x, y, color = 'gray')
        plt.title(titulo)
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15, 
                         self.correlacion_var.index[i], 
                         color = 'steelblue', ha = 'center', va = 'center')

class NoSupervisado(AnalisisDatosExploratorio):
    def __init__(self, df):
        self.__df = df 
    @property
    def df(self):
        return self.__df 
    @df.setter
    def df(self, p_df):
        self.__df = p_df

    def ACP(self, n_componentes):
        p_acp = ACPBasico(self.__df,n_componentes) 
        self.__ploteoGraficosACP(p_acp,1)
        self.__ploteoGraficosACP(p_acp,2)
        self.__ploteoGraficosACP(p_acp,3)

    def __ploteoGraficosACP(self,p_acp, tipo):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10,6), dpi = 200)
        if tipo==1:
            p_acp.plot_plano_principal()
        elif tipo==2:
            p_acp.plot_circulo()
        elif tipo==3:
            p_acp.plot_sobreposicion()   
        ax.grid(False)  
        plt.show()
    
    def HAC(self):
        p_hac = self.__df
        ward_res = ward(self.__df)      
        average_res = average(self.__df)  
        single_res  = single(self.__df)    
        complete_res = complete(self.__df)
        self.__ploteoGraficosHAC(p_hac, ward_res, 1)
        self.__ploteoGraficosHAC(p_hac, average_res, 2)
        self.__ploteoGraficosHAC(p_hac, single_res, 3)
        self.__ploteoGraficosHAC(p_hac, complete_res, 4)
        self.__clusterHAC(1)
        self.__clusterHAC(2)
        self.__clusterHAC(3)

    def __ploteoGraficosHAC(self, p_hac, res, tipo):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=200)
        if tipo == 1:
            dendrogram(res,labels = self.__df.index.tolist(), ax=ax)
            print(f"Agregación de Ward:")
        elif tipo == 2:
            dendrogram(res,labels = self.__df.index.tolist(), ax=ax)
            print(f"Salto promedio:")
        elif tipo == 3:
            dendrogram(res,labels = self.__df.index.tolist(), ax=ax)
            print(f"Salto mínimo:")
        elif tipo == 4:
            dendrogram(res,labels = self.__df.index.tolist(), ax=ax)
            print(f"Salto máximo:")
        ax.grid(False)
        plt.show()
    
    def __clusterHAC(self, tipo):
        grupos = fcluster(linkage(self.__df, method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
        grupos = grupos-1 
        centros = np.array(pd.concat([AnalisisDatosExploratorio.centroide(0, self.__df, grupos), 
                              AnalisisDatosExploratorio.centroide(1, self.__df, grupos),
                              AnalisisDatosExploratorio.centroide(2, self.__df, grupos)]))
        if tipo == 1:
            AnalisisDatosExploratorio.bar_plot(centros, self.__df.columns, scale=True)
        elif tipo ==2:
            AnalisisDatosExploratorio.bar_plot_detail(centros, self.__df.columns)
        elif tipo ==3:
            AnalisisDatosExploratorio.radar_plot(centros, self.__df.columns)
        plt.show()

    def Kmeans(self):
        self.__ploteoGraficosKMEDIAS(1)
        self.__ploteoGraficosKMEDIAS(2)

    def __ploteoGraficosKMEDIAS(self, tipo):
        if tipo == 1:
            self.__ploteoKmedias()
        elif tipo == 2:
            self.__ploteoKmedoids()
    
    def __ploteoKmedias(self):
        kmedias = KMeans(n_clusters=3, max_iter=500, n_init=150)
        kmedias.fit(self.__df)
        pca = PCA(n_components=2)
        componentes = pca.fit_transform(self.__df)
        fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 200)
        colores = ['red', 'green', 'blue']
        colores_puntos = [colores[label] for label in kmedias.predict(self.__df)]
        ax.scatter(componentes[:, 0], componentes[:, 1],c=colores_puntos)
        ax.set_xlabel('componente 1')
        ax.set_ylabel('componente 2')
        ax.set_title('3 Cluster K-Medias')
        ax.grid(False)
        plt.show()

        centros = np.array(kmedias.cluster_centers_)
        AnalisisDatosExploratorio.bar_plot(centros, self.__df.columns)
        AnalisisDatosExploratorio.bar_plot_detail(centros, self.__df.columns)
        AnalisisDatosExploratorio.radar_plot(centros, self.__df.columns)
        plt.show()

    def __ploteoKmedoids(self):
        kmedoids = KMedoids(n_clusters=3, max_iter=500, metric='cityblock')
        kmedoids.fit(self.__df)
        pca = PCA(n_components=2)
        componentes = pca.fit_transform(self.__df)
        fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 200)
        colores = ['red', 'green', 'blue']
        colores_puntos = [colores[label] for label in kmedoids.predict(self.__df)]
        ax.scatter(componentes[:, 0], componentes[:, 1],c=colores_puntos)
        ax.set_xlabel('componente 1')
        ax.set_ylabel('componente 2')
        ax.set_title('3 Cluster K-Medoids')
        ax.grid(False)
        plt.show()

        centros = np.array(kmedoids.cluster_centers_)
        AnalisisDatosExploratorio.bar_plot(centros, self.__df.columns)
        AnalisisDatosExploratorio.bar_plot_detail(centros, self.__df.columns)
        AnalisisDatosExploratorio.radar_plot(centros, self.__df.columns)
        plt.show()

    def TSNE(self, n_componentes):
        tsne = TSNE(n_componentes)
        componentes = tsne.fit_transform(self.__df)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
        ax.scatter(componentes[:, 0], componentes[:, 1])
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_title('T-SNE')
        ax.grid(False)
        plt.show()
    
    def UMAP(self, n_componentes, n_neighbors):
        modelo_umap = um.UMAP(n_componentes, n_neighbors)
        componentes = modelo_umap.fit_transform(self.__df)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
        ax.scatter(componentes[:, 0], componentes[:, 1])
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_title('UMAP')
        ax.grid(False)
        plt.show()

