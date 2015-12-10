from sklearn import svm
import pickle


def train_classifier(dic_train_path, ann_train_path):
    dic_train= pickle.load(open(dic_train_path, 'r'))
    ann_train= open(ann_train_path, 'r')

    array_clases= []
    array_bow= []

    #metemos en arrays las id's y los bow
    for k, v in dic_train.items():
        array_clases.append(k)
        array_bow.append(v)
    
    #tendremos algo asi:
    #array_clases= ["123-12345-123", "234-234-5230", "2342-6543-1230"]
    #array_bow= [[0,2,1,0,5],[2,5,7,1,9],[3,9,0,4,0]]

    #leyendo el annotations, cambiamos el vector de id's por la clase a la que pertenece esa id
    ann_train.seek(1)
    for line in ann_train:
        rec=line.split("\t")
        a= rec[1].split("\n")
        rec[1]= a[0]
        for i in range(0, len(array_clases)):
            if array_clases[i]==rec[0]:
                array_clases[i]=rec[1]

    # terminado esto tendremos (las posiciones de ambos vectores concuerdan, es decir, el primer bow sera de la primera clase, el segundo de la segunda...)
    #array_clases= ["ajuntament", "teatre_principal", "mnatec"]
    #array_bow= [[0,2,1,0,5],[2,5,7,1,9],[3,9,0,4,0]]



    #entrenamos 
    clf= svm.SVC()
    clf.fit(array_bow, array_clases)
    
    print clf.predict(array_bow[2])

    
if __name__ == "__main__":
    train_classifier("txt/bow_train.p", "TerrassaBuildings900/train/annotation.txt")
