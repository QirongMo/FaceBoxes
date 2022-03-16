

g = open('FDDB.txt','w')
for i in range(1,11):
    f = open('FDDB/FDDB-folds/FDDB-fold-{:02d}-ellipseList.txt'.format(i),'r')
    file = f.readlines()
    length = len(file)
    j = 0
    while(j<length):
        filepath = file[j].strip()+'.jpg'
        # import cv2
        # imgdir = 'FDDB/originalPics/'
        # img = cv2.imread(imgdir + filepath)
        g.write(filepath+'\t')
        num_faces = int(file[j+1].strip())
        g.write(str(num_faces))
        for k in range(1,num_faces+1):
            face_loc = file[j+1+k].strip().split()
            x1 = int(float(face_loc[3]) - float(face_loc[1]))
            y1 = int(float(face_loc[4]) - float(face_loc[0]))
            x2 = int(float(face_loc[3]) + float(face_loc[1]))
            y2 = int(float(face_loc[4]) + float(face_loc[0]))
            g.write('\t'+str(x1)+'\t'+str(y1)+'\t'+str(x2)+'\t'+str(y2)+'\t' + str(1))
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
        g.write('\n')
        j += num_faces+2
    f.close()
    g.flush()