// JavaScript 문서의 '동작'

console.log('hello javascript');

// 변수 선언, let, const
let a = 10;  // 변수
console.log(a);
a = 100;
console.log(a);

const PP = 'aaa';  // 상수
console.log(PP);
// PP = 200;

let arr = [10, 20, 30];  // 배열 (<- list)
console.log(arr);
console.log(arr[0]);

let obj = {name: 'John', age: 43};  // 오브젝트 (<- dict)
console.log(obj);
console.log(obj['name'], obj.name);

// 함수
function sayHello(name) {
    console.log("sayHello", name);
}

sayHello('홍길동');

const sayName = function(name){
    console.log("sayName", name);
}

sayName('둘리');

// class 를 사용하여 object 정의
class Alpha {
    name = 'no name';
    age = 0;

    hello() {
        console.log('hello')
    }

    hello2(greeting) {
        console.log(greeting, this.name, this.age);
    }

    // 화살표 함수
    hello3 = (bbb) => {
        return bbb + "OK!";
    }
}

let a1 = new Alpha();
console.log(a1);
console.log(a1.name, a1.age);
a1.hello();
a1.hello2('안녕하세요');
console.log(a1.hello3('파이썬'));

{
    let name = obj.name;
    let age = obj.age;
    console.log(name, age);
}

{
    // 비구조화 할당 구문
    let {name, age} = obj;
    console.log(name, age);
}

// 조건문 if
let done = true;   // boolean 타입 true/false

if (done) {
    console.log(done, '참 입니다')
} else {
    console.log(done, '거짓 입니다');
}

// 현재시간 timestamp
console.log(Date.now());


// 비동기 http 통신. fetch
let url;

url = 'https://httpbun.com/get?name=susan'
fetch(url)
.then(response => response.text())
.then(text => console.log(text))
;


async function fetch_test(name){
    let url = `https://httpbun.com/get?name=${name}`;
    response = await fetch(url);
    text = await response.text();
    console.log(text);
}

fetch_test('뽀로로');



//────────────────────────────────────────────────────────────────────
console.log("\n[프로그램 종료]", '\n'.repeat(20));