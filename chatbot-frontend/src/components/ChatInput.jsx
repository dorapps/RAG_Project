import useAutosize from '@/hooks/useAutosize';
import sendIcon from '@/assets/images/send.svg';

function ChatInput({ newMessage, isLoading, setNewMessage, submitNewMessage }) {
  const textareaRef = useAutosize(newMessage);

  function handleKeyDown(e) {
    if(e.keyCode === 13 && !e.shiftKey && !isLoading) {
      e.preventDefault();
      submitNewMessage();
    }
  }
  
  return(
    <div>
      <textarea
        ref={textareaRef}
        rows='1'
        value={newMessage}
        onChange={e => setNewMessage(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      <button onClick={submitNewMessage}>
        <img src={sendIcon} alt='send' />
      </button>
    </div>
  );
}

export default ChatInput;