import discord
import yaml

client = discord.Client()

#Type of Greeting
greeting = ['hello','hi','สวัสดี','ดี']

@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')

@client.event
async def on_message(message):
    # ignore own message
    if message.author == client.user:
        return
    if message.content == 'ping':
        return await client.send_message(message.channel, 'pong')
    if message.content.lower() in greeting :
        return await client.send_message(message.channel, 'ดี {0.author.mention}'.format(message))
    if message.content.startswith('!hello'):
        msg = 'Hello {0.author.mention}'.format(message)
        return await client.send_message(message.channel, msg)
    return await client.send_message(message.channel, "ไม่รู้อ่ะ 555")



with open('config.yml', 'r') as f:
    config = yaml.load(f)

client.run(config['token'])