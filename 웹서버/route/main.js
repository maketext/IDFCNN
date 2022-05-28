//process.env.NODE_ENV = 'production'

const events = require('events')
const express = require('express')
const app = express()
const axios = require('axios')
const port = 8888
const path = require('path')
const cors = require('cors')
const bodyParser = require('body-parser')
const fs = require('fs')
const querystring = require('querystring')
const mqtt = require('mqtt')


let baseURL = '127.0.0.1'
let mqttMsg = '.'
let mqttTopic = '/default-timer'
let isConsoleReady = 0
let client = mqtt.connect(`mqtt://${baseURL}`, {protocol:'mqtt', port:1883, clientId: 'web-server-client', qos: 0})

var wdt = new events.EventEmitter()
wdt.on('on', function() {
	if(mqttTopic == '/default-timer') return
	log(`토픽 발행=${mqttTopic}, 페이로드=${mqttMsg}`)
	client.publish(mqttTopic, mqttMsg)
})

function log(str)
{
	console.log(str)
}


app.use((req, res, next) => {
	log(`\n\n새 라우팅 ${req.path}`)

	res.on("finish", function() {
		log("응답메시지 전송됨.")
	})
	// 직렬화/역직렬화, 캐싱확인용 컴포넌트 들어갈 자리.
	next()
})

app.use(express.static(path.join(__dirname, '..', 'res')))
app.use(cors())
app.use(bodyParser.json())

axios.defaults.baseURL = "http://localhost:8888"
function patch (url, param, callback) {
	axios.patch(url, param)
		.then(function (res) {
			console.log("HTTP PATCH 응답 res.data=" + JSON.stringify(res.data))
			if(callback)
				callback(res.data)
		})
}

app.listen(port, () => {
	log("HTTP 네트워크 소켓 리스닝 중...")
	client.on('connect', () => {
		log("MQTT 네트워크 소켓 초기화됨.")
		setInterval(function () {
			wdt.emit('on')
		}, 350)

		client.subscribe('/img-start-ready', function (err) {})
		client.subscribe('/img-stop', function (err) {})

		client.subscribe('/fileopenTrain', function (err) {})
		client.subscribe('/fileopenBboxYOLO', function (err) {})
		client.subscribe('/fileopenMakePatch', function (err) {})

		client.on('message', (topic, msg) => {
			log("토픽 감지")
			switch(topic)
			{
				case '/img-start-ready':
				{
					log("/img-start-ready 토픽 감지")
					mqttTopic = '/img-recv'
					break
				}
				case '/img-stop':
				{
					mqttTopic = '/default-timer'
					break
				}
				case '/fileopenTrain':
				case '/fileopenBboxYOLO':
				case '/fileopenMakePatch':
				{
					isConsoleReady++
					console.log(topic + "++")
					break
				}
				default:
				{
				}
			}

		})
	})

	
}) 


app.set('view engine', 'pug')
app.set('views', path.join(__dirname,'..', 'pug'))


app.get('/favicon.ico', (req, res, next) => { //favicon.ico 404, 302 리다이렉트 오류나 경고 뜨면 브라우저 로딩 속도 느려진다.
	log("파비콘")
	//res.sendFile('/img/favicon.png')
	res.redirect('/img/favicon.png')
	next()
})
app.post('/isConsoleReady', (req, res, next) => {
	if(isConsoleReady >= 3)
		res.status(200).send()
	else
		res.sendStatus(404)
	next()
})
app.all("/error", (req, res, next) => {
	res.render('error', {axiosAddr: 'http://127.0.0.1:8888'})
	next()
})
app.get(['/:pugFileName', '/'], (req, res, next) => {
	log(`req.params.pugFileName=${req.params.pugFileName}`)
	if(!['index', 'edl-v10', 'epl-v20'].join().includes(req.params.pugFileName)) next()
	log(`view rendering - ${req.params.pugFileName}.pug`)
	res.render(`${req.params.pugFileName}`, {axiosAddr: baseURL})
	next()
})
let common = path.join(__dirname, '..', "res/img/")

app.get('/img/:img', (req, res, next) => {
	let im = querystring.unescape(req.params.img)
	common = path.join(__dirname, '..', "res/img/")
	let newfile = `${common}${im}.jpg`
	console.log(`im=${im}`)

	if(im == 'error') 
	{
		res.writeHead(200)
		res.end('')
	}

	log("그림 다운로드")
	log(newfile)
	fs.readFile(newfile, (err, data) => {
		if(err)
		{
			log(err)
			fs.readFile(newfile.replace('jpg', 'png'), (err, data) => {
				if(err)
				{
					fs.readFile(`${common}x.jpg`, (err, data) => {
						res.writeHead(200, {"Content-Type": "img/jpeg"})
						res.end(data)
					})
				}
				else
				{
					res.writeHead(200, {"Content-Type": "img/png"})
					res.end(data)
				}
			})

		}
		else
		{
			res.writeHead(200, {"Content-Type": "img/jpeg"})
			res.end(data)
		}
		
	})
})



/*
fs.watchFile(newfile, (curr, prev) => {
	log("파일 변경 감지")
	log(curr.ctime)
	let old = `${common}얼굴-${prev.ctime}.png`
	let old2 = `${common}얼굴.png`
	newfile = `${common}얼굴-${curr.ctime}.png`
	fs.rename(old, newfile, () => {
		log("renamed")
	})
	fs.rename(old2, newfile, () => {
	log("renamed")
	})
	
})
*/