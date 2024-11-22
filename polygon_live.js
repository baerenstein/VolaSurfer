const WebSocket = require('ws')
const APIKEY = 'UgOS5zZnpBQfCjbXRTLsontpgPB7FnLr'
// const ws = new WebSocket('wss://delayed.polygon.io/stocks') // 15-min delay
const ws = new WebSocket('wss://socket.polygon.io/stocks') // real-time

// Connection Opened:
ws.on('open', () => {
	console.log('Connected!')
	ws.send(`{"action":"auth","params":"${APIKEY}"}`)

	// aggregates
	//ws.send(`{"action":"subscribe","params":"AM.*"}`) // min
	ws.send(`{"action":"subscribe","params":"A.*"}`) // sec

	// trades
	//ws.send(`{"action":"subscribe","params":"T.*"}`)
	//ws.send(`{"action":"subscribe","params":"T.TSLA"}`)

	// quotes
	//ws.send(`{"action":"subscribe","params":"Q.*"}`)
	//ws.send(`{"action":"subscribe","params":"Q.TSLA"}`)
})

// Per message packet:
ws.on('message', ( data ) => {
	data = JSON.parse( data )
	data.map(( msg ) => {
		if( msg.ev === 'status' ){
			return console.log('Status Update:', msg.message)
		}
		console.log(msg)
	})
})

ws.on('error', console.log)