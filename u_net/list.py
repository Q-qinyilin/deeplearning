import os

def make_dataset(root):
    imgs=[]
    n=len(os.listdir(root))
    for i in range(n):
        # mask=os.path.join(root,"%03d_mask.png"%i)
        mask = os.path.join(root, "%03d_pred.png" % i)
        imgs.append(mask)
        imgs.append("\n")
    return imgs

img=make_dataset("predict/pred")
f=open("predict/pred.txt","w")
f.writelines(img)
f.close()
