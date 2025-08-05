# streamlit/components/live_feed.py
import streamlit as st
from kafka import KafkaConsumer
import json
import queue
import threading
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def show_live_feed():
    st.title("🔴 Live Recommendations Feed")
    
    # Initialize session state
    if 'recommendations_list' not in st.session_state:
        st.session_state.recommendations_list = []
        st.session_state.consumer_running = False
        st.session_state.message_count = 0
        st.session_state.stop_consumer = False
        st.session_state.message_queue = queue.Queue()
        st.session_state.raw_messages = []  # Store raw messages for debugging

    def process_queued_messages():
        """Process messages from the queue"""
        messages_processed = 0
        while not st.session_state.message_queue.empty() and messages_processed < 10:
            try:
                message_data = st.session_state.message_queue.get_nowait()
                
                # Store raw message for visibility
                st.session_state.raw_messages.insert(0, message_data['raw'])
                if len(st.session_state.raw_messages) > 20:
                    st.session_state.raw_messages = st.session_state.raw_messages[:20]
                
                # Process recommendation if valid
                if message_data['parsed']:
                    if len(st.session_state.recommendations_list) >= 50:
                        st.session_state.recommendations_list.pop()
                    st.session_state.recommendations_list.insert(0, message_data['parsed'])
                
                st.session_state.message_count += 1
                messages_processed += 1
                
            except queue.Empty:
                break
        
        return messages_processed

    def kafka_consumer_worker(offset_reset='latest'):
        """Background Kafka consumer"""
        try:
            logger.info(f"Starting consumer with offset_reset={offset_reset}")
            
            consumer = KafkaConsumer(
                'recommendations',
                bootstrap_servers=['localhost:9092'],
                auto_offset_reset=offset_reset,
                consumer_timeout_ms=2000,
                group_id=f'streamlit-live-feed-{int(time.time())}',
                enable_auto_commit=True,
                # Don't deserialize here - we want to see raw messages
                value_deserializer=None  
            )
            
            # Force assignment and log partitions
            consumer.subscribe(['recommendations'])
            consumer.poll(0)  # Force assignment
            assigned = consumer.assignment()
            logger.info(f"Assigned partitions: {assigned}")
            
            # Get current position
            for partition in assigned:
                position = consumer.position(partition)
                logger.info(f"Starting position for {partition}: {position}")
            
            poll_count = 0
            while not st.session_state.stop_consumer:
                try:
                    poll_count += 1
                    logger.info(f"Poll #{poll_count}")
                    
                    message_batch = consumer.poll(timeout_ms=1000)
                    
                    if message_batch:
                        logger.info(f"Received batch with {sum(len(msgs) for msgs in message_batch.values())} messages")
                    else:
                        logger.info("No messages in this poll")
                    
                    for topic_partition, messages in message_batch.items():
                        logger.info(f"Processing {len(messages)} messages from {topic_partition}")
                        for message in messages:
                            if st.session_state.stop_consumer:
                                break
                            
                            # Capture raw message info
                            raw_message = {
                                'offset': message.offset,
                                'partition': message.partition,
                                'timestamp': message.timestamp,
                                'key': message.key.decode('utf-8') if message.key else None,
                                'value_raw': message.value.decode('utf-8') if message.value else '',
                                'value_bytes': len(message.value) if message.value else 0,
                                'received_at': datetime.now()
                            }
                            
                            # Try to parse as JSON
                            parsed_data = None
                            parse_error = None
                            
                            try:
                                if message.value:
                                    parsed_data = {
                                        'data': json.loads(message.value.decode('utf-8')),
                                        'received_at': datetime.now(),
                                        'id': f"{message.offset}_{message.timestamp}",
                                        'offset': message.offset,
                                        'partition': message.partition
                                    }
                            except json.JSONDecodeError as e:
                                parse_error = str(e)
                            except Exception as e:
                                parse_error = f"Decode error: {str(e)}"
                            
                            raw_message['parse_error'] = parse_error
                            raw_message['parsed_successfully'] = parsed_data is not None
                            
                            # Queue both raw and parsed data
                            message_data = {
                                'raw': raw_message,
                                'parsed': parsed_data
                            }
                            
                            try:
                                st.session_state.message_queue.put(message_data, timeout=1)
                            except queue.Full:
                                logger.warning("Message queue full")
                                
                except Exception as e:
                    if not st.session_state.stop_consumer:
                        logger.error(f"Error polling: {e}")
                        time.sleep(1)
                        
            consumer.close()
            
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            st.session_state.consumer_running = False

    # Process any queued messages
    if st.session_state.consumer_running:
        processed = process_queued_messages()
        if processed > 0:
            st.success(f"📨 Processed {processed} new messages!")

    # Diagnostic tools first
    st.subheader("🔍 Diagnostics")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧪 Test Kafka Connection"):
            try:
                test_consumer = KafkaConsumer(bootstrap_servers=['localhost:9092'], consumer_timeout_ms=5000)
                topics = test_consumer.topics()
                test_consumer.close()
                st.success(f"✅ Connected! Topics: {list(topics)}")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
    
    with col2:
        if st.button("📊 Count Messages in Topic"):
            try:
                counter = KafkaConsumer(
                    'recommendations',
                    bootstrap_servers=['localhost:9092'],
                    auto_offset_reset='earliest',
                    consumer_timeout_ms=10000
                )
                
                message_count = 0
                sample_msg = None
                
                for message in counter:
                    message_count += 1
                    if message_count == 1:
                        sample_msg = {
                            'offset': message.offset,
                            'value_bytes': len(message.value) if message.value else 0,
                            'value_preview': message.value.decode('utf-8')[:100] if message.value else "EMPTY"
                        }
                    if message_count >= 100:  # Limit to avoid hanging
                        break
                
                counter.close()
                
                if message_count > 0:
                    st.success(f"✅ Found {message_count}+ messages")
                    st.json(sample_msg)
                else:
                    st.warning("⚠️ No messages found - topic is empty")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # Simple controls
    st.subheader("🎛️ Controls")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔌 Start (New Messages)") and not st.session_state.consumer_running:
            st.session_state.stop_consumer = False
            st.session_state.consumer_running = True
            thread = threading.Thread(target=lambda: kafka_consumer_worker('latest'), daemon=True)
            thread.start()
            st.success("Started consumer for new messages")
            time.sleep(1)
            st.rerun()

    with col2:
        if st.button("📜 Start (All Messages)") and not st.session_state.consumer_running:
            st.session_state.stop_consumer = False
            st.session_state.consumer_running = True
            thread = threading.Thread(target=lambda: kafka_consumer_worker('earliest'), daemon=True)
            thread.start()
            st.success("Started consumer for all messages")
            time.sleep(1)
            st.rerun()

    with col3:
        if st.button("🛑 Stop"):
            st.session_state.stop_consumer = True
            st.session_state.consumer_running = False
            st.success("Stopped consumer")
            time.sleep(1)
            st.rerun()

    # Status
    status = "🟢 Running" if st.session_state.consumer_running else "🔴 Stopped"
    st.metric("Status", status)
    st.metric("Total Messages", st.session_state.message_count)

    # Message visibility section
    st.divider()
    st.subheader("👁️ Raw Message Inspection")
    
    if st.session_state.raw_messages:
        # Summary stats
        total_msgs = len(st.session_state.raw_messages)
        successful_parses = sum(1 for msg in st.session_state.raw_messages if msg['parsed_successfully'])
        failed_parses = total_msgs - successful_parses
        empty_messages = sum(1 for msg in st.session_state.raw_messages if msg['value_bytes'] == 0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Recent Messages", total_msgs)
        with col2:
            st.metric("Successfully Parsed", successful_parses)
        with col3:
            st.metric("Parse Failures", failed_parses)
        with col4:
            st.metric("Empty Messages", empty_messages)
        
        # Show raw messages
        st.write("**Recent Raw Messages:**")
        for i, msg in enumerate(st.session_state.raw_messages[:10]):
            with st.expander(
                f"Message {i+1} - "
                f"{'✅' if msg['parsed_successfully'] else '❌'} "
                f"P{msg['partition']}:O{msg['offset']} "
                f"({msg['value_bytes']} bytes) "
                f"{msg['received_at'].strftime('%H:%M:%S')}"
            ):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Metadata:**")
                    st.json({
                        'offset': msg['offset'],
                        'partition': msg['partition'],
                        'timestamp': msg['timestamp'],
                        'key': msg['key'],
                        'value_bytes': msg['value_bytes'],
                        'parsed_successfully': msg['parsed_successfully']
                    })
                
                with col2:
                    st.write("**Raw Value:**")
                    if msg['value_bytes'] == 0:
                        st.warning("⚠️ EMPTY MESSAGE")
                    else:
                        st.code(msg['value_raw'][:500] + ("..." if len(msg['value_raw']) > 500 else ""))
                    
                    if msg['parse_error']:
                        st.error(f"Parse Error: {msg['parse_error']}")
    else:
        st.info("No messages received yet. Start the consumer to see messages.")

    # Recommendations display (only successful ones)
    st.divider()
    st.subheader("📦 Parsed Recommendations")
    
    if st.session_state.recommendations_list:
        st.success(f"Showing {len(st.session_state.recommendations_list)} parsed recommendations")
        
        for idx, rec_item in enumerate(st.session_state.recommendations_list[:5], 1):
            rec = rec_item['data']
            received_time = rec_item['received_at']
            
            # Handle both userId and user_id formats
            user_id = rec.get('user_id') or rec.get('userId', 'Unknown')
            
            with st.expander(
                f"🔥 #{idx} - User: {user_id} "
                f"({len(rec.get('recommendations', []))} products) - "
                f"{received_time.strftime('%H:%M:%S')}"
            ):
                # Show recommendations
                recommendations = rec.get('recommendations', [])
                if recommendations:
                    for i, prod in enumerate(recommendations[:5], 1):
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col1:
                            st.write(f"**{i}.**")
                        with col2:
                            st.code(prod.get('product_id', 'Unknown'))
                        with col3:
                            st.write(f"**{prod.get('score', 0):.3f}**")
                
                with st.expander("📄 Full JSON"):
                    st.json(rec)
    else:
        st.info("No parsed recommendations yet")

    # Auto-refresh
    if st.checkbox("🔄 Auto-refresh (3s)", value=True) and st.session_state.consumer_running:
        time.sleep(3)
        st.rerun()