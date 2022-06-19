
Vue.component('train', {
  data: function () {
    return {
      targetType: "green",
      show: true,
      snackbarContainer: document.querySelector('#toast'),
      packages: null,
      intervalId: null,
      fx: [],
      subclipStart: null,
      subclipEnd: null
    }
  },
  props: {
    currentProject: null
  },
  methods: {
    srcChanged(event){
        var file = event.target.files[0]
        document.getElementById('srcTxt').textContent="Source video: "+file.name+" ("+file.size+")";
    },
    targetChange(event){
        console.log(event);
        document.getElementById('srcTxt')

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
        formData.append('targetType', this.targetType);

        if(this.targetType === "color"){
            console.log(this.$refs.target_color.value);
            formData.append('target', this.$refs.target_color.value);
        }
        else if(this.targetType === "image"){
            formData.append('target', this.$refs.target_bgr_img.files[0]);
        }
        else if(this.targetType === "video"){
            formData.append('target', this.$refs.target_bgr_video.files[0]);
        }


        formData.append('fx', this.fx);
        if(this.fx.includes("subclip")) {
            formData.append('subclipStart', this.subclipStart);
            formData.append('subclipEnd', this.subclipEnd);
        }

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
            <!--<img src="/static/logo.png" width="270px" />-->
        </div>
        <!--<div class="mdl-card__supporting-text">
            Upload image file.
        </div>-->


        <div class="mdl-card__actions mdl-card--border">
            <h5>INPUT:</h5>
            <center>
                <input type="file" ref="src" @change="srcChanged" class="mybtn" id="video-source" />
                <label id="video-source-label" for="video-source" class="mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect">
                  Upload source video
                </label>

                <div class="mdl-tooltip mdl-tooltip--right mdl-tooltip--large" data-mdl-for="video-source-label">
                    Upload main video source which you want to matte.
                </div>

                <br/>
                <br/>
            </center>

            <br/>
            <br/>

            <div id="srcTxt" class="upload-btn-wrapper">Sorce video:</div>
            
            <br/>
            <br/>

            <hr/>

            <h5>BACKGROUND TARGET:</h5>
            <br/>

            <label class="mdl-radio mdl-js-radio mdl-js-ripple-effect" for="option-1">
              <input type="radio" id="option-1" v-model="targetType" class="mdl-radio__button" name="options" value="green" checked>
              <span class="mdl-radio__label">Green screen</span>
            </label>
            <br/>
            <br/>
            <label class="mdl-radio mdl-js-radio mdl-js-ripple-effect" for="option-2">
              <input type="radio" id="option-2" v-model="targetType" class="mdl-radio__button" name="options" value="color">
              <span class="mdl-radio__label">Color background:</span>
              &nbsp;&nbsp;&nbsp;&nbsp;
              <input type="color" id="colorpicker" ref="target_color" value="#ffffff">
            </label>
            <br/>
            <br/>
            <label class="mdl-radio mdl-js-radio mdl-js-ripple-effect" for="option-3">
              <input type="radio" id="option-3" v-model="targetType" class="mdl-radio__button" name="options" value="image">
              <span class="mdl-radio__label">Background image:</span>
                <input type="file" ref="target_bgr_img" class="mybtn" id="target-image-source" />
                <label id="target-image-label" for="target-image-source" class="mdl-button mdl-js-button mdl-button--primary">
                  Target image
                </label>

            </label>
            <br/>
            <br/>
            <label class="mdl-radio mdl-js-radio mdl-js-ripple-effect" for="option-4">
              <input type="radio" id="option-4" v-model="targetType" class="mdl-radio__button" name="options" value="video">
              <span class="mdl-radio__label">Background video:</span>
                &nbsp;
                <input type="file" ref="target_bgr_video" class="mybtn" id="target-video-source" />
                <label id="target-video-label" for="target-video-source" class="mdl-button mdl-js-button mdl-button--primary">
                  Target video
                </label>

            </label>
            <br/>
            <br/>

<!--
            <hr/>
            <h5>TRANSFORM:</h5>

            <input type="checkbox" id="subclip" value="subclip" v-model="fx" class="mdl-checkbox__input">
            <label for="subclip">Subclip: </label>
            <div class="mdl-textfield mdl-js-textfield mdl-textfield--floating-label">
                <input class="mdl-textfield__input" v-model="subclipStart" type="text" pattern="-?[0-9]*(\.[0-9]+)?" id="subclip_start">
                <label class="mdl-textfield__label" for="subclip_start">start (seconds)...</label>
                <span class="mdl-textfield__error">Input is not a number!</span>
            </div>
            <div class="mdl-textfield mdl-js-textfield mdl-textfield--floating-label">
                <input class="mdl-textfield__input" v-model="subclipEnd" type="text" pattern="-?[0-9]*(\.[0-9]+)?" id="subclip_end">
                <label class="mdl-textfield__label" for="subclip_end">end (seconds)...</label>
                <span class="mdl-textfield__error">Input is not a number!</span>
            </div>
            <br/>
            <input type="checkbox" id="resize" value="resize" v-model="fx" class="mdl-checkbox__input">
            <label for="subclip">Resize: </label>
            <br/>
            <input type="checkbox" id="corp" value="crop" v-model="fx" class="mdl-checkbox__input">
            <label for="subclip">Crop: </label>

-->

            <hr/>
            <input id="submit" type="submit" class="mybtn" v-on:click="submitFiles" >
            <label for="submit" class="mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect">
              Submit
            </label>
           
            
            <br/>
            <br/>
            <div id="logTxt" class="upload-btn-wrapper"></div>

        </div>
    </div>
    </div>
</main>
  `
});
