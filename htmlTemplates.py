css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://bukkuzon.com/cdn/shop/products/9781506727431_manga-cat-plus-gamer-volume-3-primary_12864d00-bcfb-4c58-b40b-40f06c5a89d2_540x.jpg?v=1690722978" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://bukkuzon.com/cdn/shop/products/9781974737451_manga-fly-me-to-the-moon-volume-19-primary_540x.jpg?v=1690723059">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''