Vue.component("dropzone",{
  template: `<div class='dropzone'></div>`,
  props: {
    currentProject: null,
  },
  data() {
    return {
      uploadDropzone: null,
      modelName: Date.now().toString()
    };
  },
  methods: {
      reload(){
      }
  },
  mounted(){
    this.uploadDropzone= new Dropzone(this.$el, {
        url:"/api/matte/"+this.modelName, 
        paramName: "files",
        method: "post",
        uploadMultiple: true,
        timeout: 36000000,
        responseType: 'arraybuffer',
        success: function(file, response){
            console.log(file)
            var imageBlob = response;
            var imageBytes = btoa(
              new Uint8Array(response)
                .reduce((data, byte) => data + String.fromCharCode(byte), '')
            );

                var outputImg = document.getElementById('output');
                outputImg.src = 'data:image/png;base64,'+imageBytes;

                var inputImg = document.getElementById('input');
                inputImg.src = file.dataURL;

                //var blob = new Blob(new Uint8Array(response), {type: "image/png"});
                //saveAs(blob, 'out.png');


        }
    });
  }
})

Vue.component('train', {
  data: function () {
    return {
      show: true,
      snackbarContainer: document.querySelector('#toast'),
      packages: null,
      intervalId: null
    }
  },
  props: {
    currentProject: null
  },
  methods: {
    train_info(){
	    axios.get("/api/training").then(response => {
            this.packages=response.data;
	    });
    },
    download(modelName) {
        console.log(modelName);
	    axios.get("/api/training/"+modelName,{
            responseType: 'arraybuffer'
        }).then(response => {
            var blob=new Blob([response.data])
            console.log(blob);
            saveAs(blob,'trt_graph.pb');
	    });
    },
    srcChanged(event){
        var file = event.target.files[0]
        document.getElementById('srcTxt').textContent="Source video: "+file.name+" ("+file.size+")";
    },
    bgrChanged(event){
        var file = event.target.files[0]
        document.getElementById('bgrTxt').textContent="Background image: "+file.name+" ("+file.size+")";

    },
    handleEvent(message) {
        console.log(message.data);
        document.getElementById('logTxt').textContent = message.data;
    },
    submitFiles() {
        this.ws = new WebSocket("ws://"+window.location.host+"/ws");
        this.ws.onmessage = this.handleEvent;

        let formData = new FormData();

        formData.append('src', this.$refs.src.files[0]);
        formData.append('bgr', this.$refs.bgr.files[0]);

        axios.post('/api/matte', formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            },
            responseType: 'arraybuffer'
          }
        ).then(response => {
            var blob=new Blob([response.data])
            console.log(blob);
            saveAs(blob,'output.mp4');
	    })
        .catch(function(){
        });
    }
  },
  created(){
      //this.intervalId = setInterval(this.train_info, 5000);
      //  axios.get("/static/woman.jpg",{
      //        responseType: 'arraybuffer'
      //    }).then(response => {
      //        var blob=new Blob([response.data])
      //        console.log(blob);
      //        var formData = new FormData();
      //        formData.append("file", blob);
      //        axios.post('/api/scissors/333', formData, {
      //            headers: {
      //              'Content-Type': 'multipart/form-data'
      //            }, responseType: 'arraybuffer'

      //        }).then(res => {
      //            console.log(res);

      //        var imageBlob = res.data;
      //        var imageBytes = btoa(
      //          new Uint8Array(res.data)
      //            .reduce((data, byte) => data + String.fromCharCode(byte), '')
      //        );

      //            var outputImg = document.getElementById('output');
      //            outputImg.src = 'data:image/png;base64,'+imageBytes;

      //        });
	  //    });
     
  },
  updated(){
      if(this.$refs.dropzone !== undefined){
        this.$refs.dropzone.reload(this.currentProject);
      }
  },
  template: `
  <main class="mdl-layout__content mdl-color--grey-100" v-if="show">
  <div class="mdl-grid demo-content">
    <div class="demo-card-square mdl-card mdl-cell mdl-cell--12-col">
        <div class="mdl-card__title mdl-card--expand">
            <!--<h2 class="mdl-card__title-text">AI Scissors</h2>-->
            <img src="/static/logo.png" width="270px" />
        </div>
        <!--<div class="mdl-card__supporting-text">
            Upload image file.
        </div>-->


        <div class="mdl-card__actions mdl-card--border">
            <!-- <dropzone :current-project="currentProject" ref="dropzone"></dropzone> -->
            <br/>
            <div class="upload-btn-wrapper">
                <button class="btn">Upload source</button>
                <input type="file" ref="src" @change="srcChanged" name="myfile" />
            </div>
            <div class="upload-btn-wrapper">
                <button class="btn">Upload background</button>
                <input type="file" ref="bgr" @change="bgrChanged" name="myfile" />
            </div>
            <br/>
            <br/>

            <div id="srcTxt" class="upload-btn-wrapper">Sorce video:</div>
            <br/>
            <div id="bgrTxt" class="upload-btn-wrapper">Background image:</div>
            
            <br/>
            <br/>

            <input class="btn" type="submit" v-on:click="submitFiles" >
            <br/>
            <br/>
            <div id="logTxt" class="upload-btn-wrapper"></div>
        </div>
    </div>
    </div>
</main>
  `
});
