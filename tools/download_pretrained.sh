mkdir -p pretrained && \
cd pretrained && \
wget -O bevfusion-det.pth https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v && \
wget -O bevfusion-seg.pth https://www.dropbox.com/scl/fi/8lgd1hkod2a15mwry0fvd/bevfusion-seg.pth?rlkey=2tmgw7mcrlwy9qoqeui63tay9 && \
wget -O lidar-only-det.pth https://www.dropbox.com/scl/fi/b1zvgrg9ucmv0wtx6pari/lidar-only-det.pth?rlkey=fw73bmdh57jxtudw6osloywah && \
wget -O lidar-only-seg.pth https://www.dropbox.com/scl/fi/mi3w6uxvytdre9i42r9k7/lidar-only-seg.pth?rlkey=rve7hx80u3en1gfoi7tjucl72 && \
wget -O camera-only-det.pth https://www.dropbox.com/scl/fi/pxfaz1nc07qa2twlatzkz/camera-only-det.pth?rlkey=f5do81fawie0ssbg9uhrm6p30 && \
wget -O camera-only-seg.pth https://www.dropbox.com/scl/fi/cwpcu80n0shmwraegi6z4/camera-only-seg.pth?rlkey=l60kdaz19fq3gwocsjk09e60z && \
wget -O swint-nuimages-pretrained.pth https://www.dropbox.com/scl/fi/f3e67wgn2omoftah4ceri/swint-nuimages-pretrained.pth?rlkey=k9kafympye80b3b1quutti4yq
