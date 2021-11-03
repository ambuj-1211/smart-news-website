const spawn = require('child_process').spawn;
let text = 'New species of alien have been found on planet Cheemsland. The human race will have a hard time naming the new species as they have only one testicle.';
const childPython = spawn('python3', ['test.py', text]);

childPython.stdout.on('data', (data) => {
	console.log('Output from Python:', data.toString());
});
