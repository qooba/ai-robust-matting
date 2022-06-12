app=new Vue({ 
  el: '#app',
  data: {
    ws: null
  },
  methods: {
    train() {
        this.$refs.train.show=true;
    },
    handleEvent(event) {
        data=JSON.parse(event.data);
    }
  },
  created: function () {
  }
});
