
# coding: utf-8

# In[ ]:


from torch.utils.data import Dataset, DataLoader
class Dataset(Dataset):
    def __init__(self,train_x,train_y):
        self.train_x=train_x
        self.train_y=train_y
        self.data_len = len(train_x)

    def __getitem__(self, index):

        train_x=self.train_x[index]
        train_y=self.train_y[index]
        return (train_x, train_y)
    def __len__(self):
        return self.data_len
    
class dataset_dataloader:
    def __init__ (self,is_whole,is_cross,method,cross_number_for_test=1,batchsize=24,numworkers=2):
        if is_cross==False:
            import pandas as pd
            import numpy as np
            from PIL import Image
            from pyts.visualization import plot_gasf
            import matplotlib.pyplot as plt
            import torch
            from torch.autograd import Variable

            if is_whole==True:
                whole="_all"
            else:
                whole=""

            normal   = pd.read_json("normal_train"+whole+".json")
            abnormal = pd.read_json("abnormal_train"+whole+".json")
            x=normal.append(abnormal)
            x=x.sample(frac=1)

            def pil_loader(path):
                with open(path, 'rb') as f:
                    img = Image.open(f)
                    return img.convert('L')
            lst = list()
            count=0
            lst2=list()
            y=list()

            for i in x.batch.unique():
                df=x[x.batch==i]
                lst = list()
                if np.all(df.status=="abnormal"):
                    path="wafer_img"+whole+"/gadf/train/abnormal/"
                    path_2="wafer_img"+whole+"/mtf/train/abnormal/"
                    path_3="wafer_img"+whole+"/gasf/train/abnormal/"
                    path_4="wafer_img"+whole+"/recurrence_plots/train/abnormal/"
                else:
                    path="wafer_img"+whole+"/gadf/train/normal/"
                    path_2="wafer_img"+whole+"/mtf/train/normal/"
                    path_3="wafer_img"+whole+"/gasf/train/normal/"
                    path_4="wafer_img"+whole+"/recurrence_plots/train/normal/"

                list_im = [
                    path+str(i)+'_6.png', 
                    path+str(i)+'_7.png', 
                    path+str(i)+'_8.png', 
                    path+str(i)+'_11.png', 
                    path+str(i)+'_12.png', 
                    path+str(i)+'_15.png',      
                ]
                list_im_2 = [
                    path+str(i)+'_6.png', 
                    path+str(i)+'_7.png', 
                    path+str(i)+'_8.png', 
                    path+str(i)+'_11.png', 
                    path+str(i)+'_12.png', 
                    path+str(i)+'_15.png',      
                ]
                list_im_3 = [
                    path+str(i)+'_6.png', 
                    path+str(i)+'_7.png', 
                    path+str(i)+'_8.png', 
                    path+str(i)+'_11.png', 
                    path+str(i)+'_12.png', 
                    path+str(i)+'_15.png',      
                ]
                list_im_4 = [
                    path+str(i)+'_6.png', 
                    path+str(i)+'_7.png', 
                    path+str(i)+'_8.png', 
                    path+str(i)+'_11.png', 
                    path+str(i)+'_12.png', 
                    path+str(i)+'_15.png',      
                ]
                
                imgs = [ pil_loader(j) for j in list_im ]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs])[0][1]
                imgs_comb = np.vstack( (np.asarray( j.resize(min_shape) ) for j in imgs ) )
                imgs_comb = Image.fromarray( imgs_comb)

                imgs_2 = [ pil_loader(j) for j in list_im_2 ]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs_2])[0][1]
                imgs_comb_2 = np.vstack( (np.asarray( j.resize(min_shape) ) for j in imgs_2 ) )
                imgs_comb_2 = Image.fromarray( imgs_comb_2)

                imgs_3 = [ pil_loader(j) for j in list_im_3 ]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs_3])[0][1]
                imgs_comb_3 = np.vstack( (np.asarray( j.resize(min_shape) ) for j in imgs_3 ) )
                imgs_comb_3 = Image.fromarray( imgs_comb_3)

                imgs_4 = [ pil_loader(j) for j in list_im_4 ]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs_4])[0][1]
                imgs_comb_4 = np.vstack( (np.asarray( j.resize(min_shape) ) for j in imgs_4 ) )
                imgs_comb_4 = Image.fromarray( imgs_comb_4)

                imgs_5=[imgs_comb,imgs_comb_2,imgs_comb_3,imgs_comb_4]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs_5])[0][1]
                imgs_comb_5 = np.hstack( (np.asarray( j.resize(min_shape) ) for j in imgs_5 ) )
                imgs_comb_5 = Image.fromarray( imgs_comb_5)

                lst.append(np.array(imgs_comb_5)/255)

                arr = np.array(lst)
                if df.iloc[0].status=="normal":
                    y.append(0)
                else:
                    y.append(1)

                lst2.append(arr)    
            #     count+=1
            #     if count==1:
            #         break
            train_x = torch.from_numpy(np.array(lst2)).float()
            train_y = torch.LongTensor(np.array(y))

            train_data = Dataset(train_x,train_y)
            train_loader = DataLoader(dataset=train_data,
                                      batch_size=24,
                                      shuffle=True,
                                      num_workers=2)
            self.train_x=train_x
            self.train_y=train_y
            self.train_loader=train_loader

            #test data set
            normal   = pd.read_json("normal_validation"+whole+".json")
            abnormal = pd.read_json("abnormal_validation"+whole+".json")
            x=normal.append(abnormal)
            x=x.sample(frac=1)

            # print(x)
            lst = list()
            count=0
            lst2=list()
            y=list()
            path=""
            for i in x.batch.unique():
                df=x[x.batch==i]
                lst = list()
                if np.all(df.status=="abnormal"):
                    path="wafer_img"+whole+"/gadf/validation/abnormal/"
                    path_2="wafer_img"+whole+"/mtf/validation/abnormal/"
                    path_3="wafer_img"+whole+"/gasf/validation/abnormal/"
                    path_4="wafer_img"+whole+"/recurrence_plots/validation/abnormal/"
                else:
                    path="wafer_img"+whole+"/gadf/validation/normal/"
                    path_2="wafer_img"+whole+"/mtf/validation/normal/"
                    path_3="wafer_img"+whole+"/gasf/validation/normal/"
                    path_4="wafer_img"+whole+"/recurrence_plots/validation/normal/"

                list_im = [
                    path+str(i)+'_6.png', 
                    path+str(i)+'_7.png', 
                    path+str(i)+'_8.png', 
                    path+str(i)+'_11.png', 
                    path+str(i)+'_12.png', 
                    path+str(i)+'_15.png',      
                ]
                list_im_2 = [
                    path+str(i)+'_6.png', 
                    path+str(i)+'_7.png', 
                    path+str(i)+'_8.png', 
                    path+str(i)+'_11.png', 
                    path+str(i)+'_12.png', 
                    path+str(i)+'_15.png',      
                ]
                list_im_3 = [
                    path+str(i)+'_6.png', 
                    path+str(i)+'_7.png', 
                    path+str(i)+'_8.png', 
                    path+str(i)+'_11.png', 
                    path+str(i)+'_12.png', 
                    path+str(i)+'_15.png',      
                ]
                list_im_4 = [
                    path+str(i)+'_6.png', 
                    path+str(i)+'_7.png', 
                    path+str(i)+'_8.png', 
                    path+str(i)+'_11.png', 
                    path+str(i)+'_12.png', 
                    path+str(i)+'_15.png',      
                ]
                
                imgs = [ pil_loader(j) for j in list_im ]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs])[0][1]
                imgs_comb = np.vstack( (np.asarray( j.resize(min_shape) ) for j in imgs ) )
                imgs_comb = Image.fromarray( imgs_comb)

                imgs_2 = [ pil_loader(j) for j in list_im_2 ]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs_2])[0][1]
                imgs_comb_2 = np.vstack( (np.asarray( j.resize(min_shape) ) for j in imgs_2 ) )
                imgs_comb_2 = Image.fromarray( imgs_comb_2)

                imgs_3 = [ pil_loader(j) for j in list_im_3 ]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs_3])[0][1]
                imgs_comb_3 = np.vstack( (np.asarray( j.resize(min_shape) ) for j in imgs_3 ) )
                imgs_comb_3 = Image.fromarray( imgs_comb_3)

                imgs_4 = [ pil_loader(j) for j in list_im_4 ]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs_4])[0][1]
                imgs_comb_4 = np.vstack( (np.asarray( j.resize(min_shape) ) for j in imgs_4 ) )
                imgs_comb_4 = Image.fromarray( imgs_comb_4)

                imgs_5=[imgs_comb,imgs_comb_2,imgs_comb_3,imgs_comb_4]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs_5])[0][1]
                imgs_comb_5 = np.hstack( (np.asarray( j.resize(min_shape) ) for j in imgs_5 ) )
                imgs_comb_5 = Image.fromarray( imgs_comb_5)

                lst.append(np.array(imgs_comb_5)/255)
                arr = np.array(lst)
                if df.iloc[0].status=="normal":
                    y.append(0)
                else:
                    y.append(1)

                lst2.append(arr)    
            #     if count==1:
            #         break
            test_x = Variable(torch.from_numpy(np.array(lst2)).float()).cuda()
            test_y = torch.LongTensor(np.array(y)).cuda()
            self.test_x=test_x
            self.test_y=test_y        

            print("done")
            
        else:
            import pandas as pd
            import numpy as np
            from PIL import Image
            import matplotlib.pyplot as plt
            import torch
            from torch.autograd import Variable

            if is_whole==True:
                whole="_all"
            else:
                whole=""
            cross=["1","2","3","4","5"]
            cross.remove(str(cross_number_for_test))
            x = [  
                pd.read_json("normal"+whole+"_cross_"+cross[0]+".json"),
                pd.read_json("normal"+whole+"_cross_"+cross[1]+".json"),
                pd.read_json("normal"+whole+"_cross_"+cross[2]+".json"),
                pd.read_json("normal"+whole+"_cross_"+cross[3]+".json"),
                pd.read_json("abnormal"+whole+"_cross_"+cross[0]+".json"),
                pd.read_json("abnormal"+whole+"_cross_"+cross[1]+".json"),
                pd.read_json("abnormal"+whole+"_cross_"+cross[2]+".json"),
                pd.read_json("abnormal"+whole+"_cross_"+cross[3]+".json")
            ]
            x = pd.concat(x)
            x =  x.sample(frac=1)


            def pil_loader(path):
                with open(path, 'rb') as f:
                    img = Image.open(f)
                    return img.convert('L')
            lst = list()
            count=0
            lst2=list()
            y=list()

            for i in x.batch.unique():
                df=x[x.batch==i]
                lst = list()
                if np.all(df.status=="abnormal"):
                    path="wafer_img_cross"+whole+"/gadf/"+str(df.iloc[0].cross)+"/abnormal/"
                    path_2="wafer_img_cross"+whole+"/gasf/"+str(df.iloc[0].cross)+"/abnormal/"
                    path_3="wafer_img_cross"+whole+"/mtf/"+str(df.iloc[0].cross)+"/abnormal/"
                    path_4="wafer_img_cross"+whole+"/recurrence_plots/"+str(df.iloc[0].cross)+"/abnormal/"
                else:
                    path="wafer_img_cross"+whole+"/gadf/"+str(df.iloc[0].cross)+"/normal/"
                    path_2="wafer_img_cross"+whole+"/gasf/"+str(df.iloc[0].cross)+"/normal/"
                    path_3="wafer_img_cross"+whole+"/mtf/"+str(df.iloc[0].cross)+"/normal/"
                    path_4="wafer_img_cross"+whole+"/recurrence_plots/"+str(df.iloc[0].cross)+"/normal/"

                list_im = [
                    path+str(i)+'_6.png', 
                    path+str(i)+'_7.png', 
                    path+str(i)+'_8.png', 
                    path+str(i)+'_11.png', 
                    path+str(i)+'_12.png', 
                    path+str(i)+'_15.png',      
                ]
                list_im_2 = [
                    path+str(i)+'_6.png', 
                    path+str(i)+'_7.png', 
                    path+str(i)+'_8.png', 
                    path+str(i)+'_11.png', 
                    path+str(i)+'_12.png', 
                    path+str(i)+'_15.png',      
                ]
                list_im_3 = [
                    path+str(i)+'_6.png', 
                    path+str(i)+'_7.png', 
                    path+str(i)+'_8.png', 
                    path+str(i)+'_11.png', 
                    path+str(i)+'_12.png', 
                    path+str(i)+'_15.png',      
                ]
                list_im_4 = [
                    path+str(i)+'_6.png', 
                    path+str(i)+'_7.png', 
                    path+str(i)+'_8.png', 
                    path+str(i)+'_11.png', 
                    path+str(i)+'_12.png', 
                    path+str(i)+'_15.png',      
                ]
                
                imgs = [ pil_loader(j) for j in list_im ]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs])[0][1]
                imgs_comb = np.vstack( (np.asarray( j.resize(min_shape) ) for j in imgs ) )
                imgs_comb = Image.fromarray( imgs_comb)

                imgs_2 = [ pil_loader(j) for j in list_im_2 ]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs_2])[0][1]
                imgs_comb_2 = np.vstack( (np.asarray( j.resize(min_shape) ) for j in imgs_2 ) )
                imgs_comb_2 = Image.fromarray( imgs_comb_2)

                imgs_3 = [ pil_loader(j) for j in list_im_3 ]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs_3])[0][1]
                imgs_comb_3 = np.vstack( (np.asarray( j.resize(min_shape) ) for j in imgs_3 ) )
                imgs_comb_3 = Image.fromarray( imgs_comb_3)

                imgs_4 = [ pil_loader(j) for j in list_im_4 ]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs_4])[0][1]
                imgs_comb_4 = np.vstack( (np.asarray( j.resize(min_shape) ) for j in imgs_4 ) )
                imgs_comb_4 = Image.fromarray( imgs_comb_4)

                imgs_5=[imgs_comb,imgs_comb_2,imgs_comb_3,imgs_comb_4]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs_5])[0][1]
                imgs_comb_5 = np.hstack( (np.asarray( j.resize(min_shape) ) for j in imgs_5 ) )
                imgs_comb_5 = Image.fromarray( imgs_comb_5)

                lst.append(np.array(imgs_comb_5)/255)

                arr = np.array(lst)
                if df.iloc[0].status=="normal":
                    y.append(0)
                else:
                    y.append(1)

                lst2.append(arr)    
            #     count+=1
            #     if count==1:
            #         break
            train_x = torch.from_numpy(np.array(lst2)).float()
            train_y = torch.LongTensor(np.array(y))

            train_data = Dataset(train_x,train_y)
            train_loader = DataLoader(dataset=train_data,
                                      batch_size=24,
                                      shuffle=True,
                                      num_workers=2)
            self.train_x=train_x
            self.train_y=train_y
            self.train_loader=train_loader

            #test data set
            normal   = pd.read_json("normal"+whole+"_cross_"+str(cross_number_for_test)+".json")
            abnormal = pd.read_json("abnormal"+whole+"_cross_"+str(cross_number_for_test)+".json")
            x=normal.append(abnormal)
            x=x.sample(frac=1)

            # print(x)
            lst = list()
            count=0
            lst2=list()
            y=list()
            path=""
            for i in x.batch.unique():
                df=x[x.batch==i]
                lst = list()
                if np.all(df.status=="abnormal"):
                    path="wafer_img_cross"+whole+"/gadf/"+str(df.iloc[0].cross)+"/abnormal/"
                    path_2="wafer_img_cross"+whole+"/gasf/"+str(df.iloc[0].cross)+"/abnormal/"
                    path_3="wafer_img_cross"+whole+"/mtf/"+str(df.iloc[0].cross)+"/abnormal/"
                    path_4="wafer_img_cross"+whole+"/recurrence_plots/"+str(df.iloc[0].cross)+"/abnormal/"
                else:
                    path="wafer_img_cross"+whole+"/gadf/"+str(df.iloc[0].cross)+"/normal/"
                    path_2="wafer_img_cross"+whole+"/gasf/"+str(df.iloc[0].cross)+"/normal/"
                    path_3="wafer_img_cross"+whole+"/mtf/"+str(df.iloc[0].cross)+"/normal/"
                    path_4="wafer_img_cross"+whole+"/recurrence_plots/"+str(df.iloc[0].cross)+"/normal/"

                list_im = [
                    path+str(i)+'_6.png', 
                    path+str(i)+'_7.png', 
                    path+str(i)+'_8.png', 
                    path+str(i)+'_11.png', 
                    path+str(i)+'_12.png', 
                    path+str(i)+'_15.png',      
                ]
                list_im_2 = [
                    path+str(i)+'_6.png', 
                    path+str(i)+'_7.png', 
                    path+str(i)+'_8.png', 
                    path+str(i)+'_11.png', 
                    path+str(i)+'_12.png', 
                    path+str(i)+'_15.png',      
                ]
                list_im_3 = [
                    path+str(i)+'_6.png', 
                    path+str(i)+'_7.png', 
                    path+str(i)+'_8.png', 
                    path+str(i)+'_11.png', 
                    path+str(i)+'_12.png', 
                    path+str(i)+'_15.png',      
                ]
                list_im_4 = [
                    path+str(i)+'_6.png', 
                    path+str(i)+'_7.png', 
                    path+str(i)+'_8.png', 
                    path+str(i)+'_11.png', 
                    path+str(i)+'_12.png', 
                    path+str(i)+'_15.png',      
                ]
                imgs = [ pil_loader(j) for j in list_im ]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs])[0][1]
                imgs_comb = np.vstack( (np.asarray( j.resize(min_shape) ) for j in imgs ) )
                imgs_comb = Image.fromarray( imgs_comb)

                imgs_2 = [ pil_loader(j) for j in list_im_2 ]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs_2])[0][1]
                imgs_comb_2 = np.vstack( (np.asarray( j.resize(min_shape) ) for j in imgs_2 ) )
                imgs_comb_2 = Image.fromarray( imgs_comb_2)

                imgs_3 = [ pil_loader(j) for j in list_im_3 ]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs_3])[0][1]
                imgs_comb_3 = np.vstack( (np.asarray( j.resize(min_shape) ) for j in imgs_3 ) )
                imgs_comb_3 = Image.fromarray( imgs_comb_3)

                imgs_4 = [ pil_loader(j) for j in list_im_4 ]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs_4])[0][1]
                imgs_comb_4 = np.vstack( (np.asarray( j.resize(min_shape) ) for j in imgs_4 ) )
                imgs_comb_4 = Image.fromarray( imgs_comb_4)

                imgs_5=[imgs_comb,imgs_comb_2,imgs_comb_3,imgs_comb_4]
                min_shape = sorted( [(np.sum(j.size), j.size ) for j in imgs_5])[0][1]
                imgs_comb_5 = np.hstack( (np.asarray( j.resize(min_shape) ) for j in imgs_5 ) )
                imgs_comb_5 = Image.fromarray( imgs_comb_5)

                lst.append(np.array(imgs_comb_5)/255)
                
                arr = np.array(lst)
                if df.iloc[0].status=="normal":
                    y.append(0)
                else:
                    y.append(1)

                lst2.append(arr)    
            #     if count==1:
            #         break
            test_x = Variable(torch.from_numpy(np.array(lst2)).float()).cuda()
            test_y = torch.LongTensor(np.array(y)).cuda()
            self.test_x=test_x
            self.test_y=test_y        

            print("done")
            
        
print("Function setting done")

