import argparse, cv2, matplotlib.pyplot as plt
def main():
    p=argparse.ArgumentParser(); p.add_argument("--image",required=True); a=p.parse_args()
    image=cv2.imread(a.image,cv2.IMREAD_GRAYSCALE)
    if image is None: raise SystemExit("Could not read image: "+a.image)
    resized=cv2.resize(image,(256,256))
    fig,ax=plt.subplots(1,2,figsize=(8,4)); ax[0].imshow(image,cmap="gray"); ax[0].set_title("Original")
    ax[1].imshow(resized,cmap="gray"); ax[1].set_title("Resized 256x256")
    for item in ax:item.axis("off")
    plt.tight_layout(); plt.show()
if __name__=="__main__":main()
